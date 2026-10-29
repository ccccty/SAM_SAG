/* SAM_SAG 前端逻辑 —— 纯原生 JS, 零依赖, 离线可用。
 *
 * 与后端契约:
 *   GET  /api/models   → {ok, models:{yolo, anomalib:{categories:[...]}, sam3}}
 *   POST /api/detect   → {ok, detections:[{label,confidence,box:[x1,y1,x2,y2]}], inference_ms}
 *   POST /api/inspect  → {ok, category, score, is_ng, threshold,
 *                         heatmap_png_base64, overlay_png_base64, inference_ms}
 *   POST /api/segment  → {ok, result:{concepts, model, device, inference_ms, image_size,
 *                         instances:[{concept,score,box,area_px,area_ratio,mask_png_base64}],
 *                         mask_png_base64, overlay_png_base64}}
 *   错误统一 {ok:false, error:{code,message}}
 */
(function () {
  'use strict';

  // ------------------------------------------------------------------
  // 状态
  // ------------------------------------------------------------------
  var HISTORY_KEY = 'sam_sag_history_v1';
  var CANVAS_MAX_WIDTH = 960;

  var state = {
    file: null,          // 当前上传的 File
    dataURL: null,       // 原图 dataURL(画布绘制用)
    mode: 'detect',      // detect | inspect | segment
    models: null,        // /api/models 返回
    lastResult: null,    // 最近一次推理结果(渲染/切视图用)
  };

  // ------------------------------------------------------------------
  // DOM
  // ------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }

  var els = {
    dropzone: $('dropzone'),
    fileInput: $('fileInput'),
    dzFile: $('dzFile'),
    categoryRow: $('categoryRow'),
    categorySelect: $('categorySelect'),
    confRow: $('confRow'),
    confSlider: $('confSlider'),
    confVal: $('confVal'),
    threshRow: $('threshRow'),
    threshSlider: $('threshSlider'),
    threshVal: $('threshVal'),
    promptRow: $('promptRow'),
    promptInput: $('promptInput'),
    segThresholdRow: $('segThresholdRow'),
    segThresholdSlider: $('segThresholdSlider'),
    segThresholdVal: $('segThresholdVal'),
    maskRow: $('maskRow'),
    runBtn: $('runBtn'),
    statusBar: $('statusBar'),
    verdict: $('verdict'),
    metaBox: $('metaBox'),
    viewRow: $('viewRow'),
    stage: $('stage'),
    emptyHint: $('emptyHint'),
    historyList: $('historyList'),
    clearHistory: $('clearHistory'),
    badgeYolo: $('badgeYolo'),
    badgeAnomalib: $('badgeAnomalib'),
    badgeSam3: $('badgeSam3'),
  };
  var ctx = els.stage.getContext('2d');

  // ------------------------------------------------------------------
  // 工具
  // ------------------------------------------------------------------
  function setStatus(msg, kind) {
    els.statusBar.textContent = msg || '';
    els.statusBar.className = 'status-bar' + (kind ? ' ' + kind : '');
  }

  function setBadge(el, name, available, message) {
    els['badge' + name].textContent = name + ' · ' + (available ? '就绪' : '未就绪');
    els['badge' + name].className = 'badge ' + (available ? 'on' : 'off');
    els['badge' + name].title = message || '';
  }

  function setVerdict(text, kind) {
    els.verdict.textContent = text;
    els.verdict.className = 'verdict verdict-' + kind;
  }

  // 视图值 → 胶囊 class; segment 只给 mask/overlay, inspect 保持 overlay/heatmap/original
  var VIEW_INSPECT = ['overlay', 'heatmap', 'original'];
  var VIEW_SEGMENT = ['mask', 'overlay'];

  function checkedView() {
    var el = document.querySelector('input[name="view"]:checked');
    return el ? el.value : 'overlay';
  }

  function setView(value) {
    var el = els.viewRow.querySelector('input[name="view"][value="' + value + '"]');
    if (el) { el.checked = true; }
  }

  function syncViewPills(mode) {
    var allowed = mode === 'segment' ? VIEW_SEGMENT : VIEW_INSPECT;
    VIEW_INSPECT.concat('mask').forEach(function (v) {
      var pill = els.viewRow.querySelector('.view-' + v);
      if (pill) { pill.classList.toggle('hidden', allowed.indexOf(v) < 0); }
    });
    if (allowed.indexOf(checkedView()) < 0) { setView('overlay'); }
  }

  function loadImage(src) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () { resolve(img); };
      img.onerror = function () { reject(new Error('图片解码失败')); };
      img.src = src;
    });
  }

  // 把图片适配画布宽度后绘制, 返回缩放比例
  function drawImageFit(img) {
    var scale = Math.min(1, CANVAS_MAX_WIDTH / img.naturalWidth);
    els.stage.width = Math.round(img.naturalWidth * scale);
    els.stage.height = Math.round(img.naturalHeight * scale);
    ctx.drawImage(img, 0, 0, els.stage.width, els.stage.height);
    return scale;
  }

  function makeThumb() {
    var thumb = document.createElement('canvas');
    var scale = Math.min(1, 160 / els.stage.width);
    thumb.width = Math.max(1, Math.round(els.stage.width * scale));
    thumb.height = Math.max(1, Math.round(els.stage.height * scale));
    thumb.getContext('2d').drawImage(els.stage, 0, 0, thumb.width, thumb.height);
    return thumb.toDataURL('image/jpeg', 0.6);
  }

  // ------------------------------------------------------------------
  // 模型状态
  // ------------------------------------------------------------------
  function refreshModels() {
    return fetch('/api/models')
      .then(function (r) { return r.json(); })
      .then(function (json) {
        if (!json.ok) { throw new Error((json.error && json.error.message) || 'models 接口异常'); }
        state.models = json.models;
        setBadge(els.badgeYolo, 'Yolo', json.models.yolo.available, json.models.yolo.message);
        setBadge(els.badgeAnomalib, 'Anomalib', json.models.anomalib.available, json.models.anomalib.message);
        setBadge(els.badgeSam3, 'Sam3', json.models.sam3.available, json.models.sam3.message);

        // SAM3 概念候选提示(可选字段, 缺失时保留 HTML 默认文案)
        var sam3 = json.models.sam3 || {};
        var hintEl = els.promptRow.querySelector('.field-hint');
        if (hintEl && sam3.concepts_hint && sam3.concepts_hint.length) {
          hintEl.textContent = '概念候选: ' + sam3.concepts_hint.join(' / ');
        }

        // 瑕疵类别下拉
        var cats = json.models.anomalib.categories || [];
        els.categorySelect.innerHTML = '';
        if (cats.length === 0) {
          var opt = document.createElement('option');
          opt.value = '';
          opt.textContent = '(无可用类别模型)';
          els.categorySelect.appendChild(opt);
        } else {
          cats.forEach(function (c) {
            var opt = document.createElement('option');
            opt.value = c.name;
            opt.textContent = c.name + ' (' + c.ckpt + ')';
            els.categorySelect.appendChild(opt);
          });
        }
        updateRunButton();
      })
      .catch(function (err) {
        setStatus('模型状态获取失败: ' + err.message, 'error');
      });
  }

  // ------------------------------------------------------------------
  // 模式与控件
  // ------------------------------------------------------------------
  function setMode(mode) {
    state.mode = mode;
    var isInspect = mode === 'inspect';
    var isSegment = mode === 'segment';
    els.categoryRow.classList.toggle('hidden', !isInspect);
    els.threshRow.classList.toggle('hidden', !isInspect);
    els.promptRow.classList.toggle('hidden', !isSegment);
    els.segThresholdRow.classList.toggle('hidden', !isSegment);
    els.confRow.classList.toggle('hidden', mode !== 'detect');
    els.viewRow.classList.add('hidden');
    els.maskRow.classList.add('hidden');
    els.maskRow.innerHTML = '';
    syncViewPills(mode);
    updateRunButton();
  }

  // SAM3 就绪判定: /api/models 的 sam3.available
  function sam3Info() {
    return (state.models && state.models.sam3) || null;
  }

  function updateRunButton() {
    var ready = !!state.file;
    var hint = '';
    if (state.mode === 'inspect') {
      ready = ready && !!els.categorySelect.value;
    } else if (state.mode === 'segment') {
      var sam3 = sam3Info();
      if (!sam3 || !sam3.available) {
        ready = false;
        hint = (sam3 && sam3.message) || 'SAM3 模型未就绪';
      } else if (!els.promptInput.value.trim()) {
        ready = false;
        hint = '请输入文本提示, 例如 bottle,broken';
      }
    }
    els.runBtn.disabled = !ready;
    els.runBtn.textContent = state.mode === 'inspect' ? '开始瑕疵检测'
      : state.mode === 'segment' ? '开始文本分割'
      : '开始目标检测';
    if (state.mode === 'segment' && hint) { setStatus(hint, 'error'); }
  }

  // ------------------------------------------------------------------
  // 文件上传
  // ------------------------------------------------------------------
  function handleFile(file) {
    if (!file) { return; }
    if (file.type && file.type.indexOf('image/') !== 0) {
      setStatus('请选择图片文件', 'error');
      return;
    }
    var reader = new FileReader();
    reader.onload = function () {
      state.file = file;
      state.dataURL = reader.result;
      state.lastResult = null;
      els.dzFile.textContent = file.name + ' (' + Math.round(file.size / 1024) + ' KB)';
      setStatus('图片已加载, 可以开始检测', 'ok');
      setVerdict('等待检测', 'idle');
      els.metaBox.textContent = '';
      els.viewRow.classList.add('hidden');
      els.maskRow.classList.add('hidden');
      els.maskRow.innerHTML = '';
      els.emptyHint.style.display = 'none';
      loadImage(state.dataURL).then(drawImageFit).catch(function () {});
      updateRunButton();
    };
    reader.readAsDataURL(file);
  }

  // ------------------------------------------------------------------
  // 推理
  // ------------------------------------------------------------------
  // /api/detect /api/inspect 的耗时在顶层; /api/segment 的耗时在 result 内
  function resultMs(json) {
    if (json.inference_ms != null) { return json.inference_ms; }
    return (json.result && json.result.inference_ms) != null ? json.result.inference_ms : '-';
  }

  function run() {
    if (!state.file) { return; }
    els.runBtn.disabled = true;
    setStatus('推理中…', '');

    var fd = new FormData();
    fd.append('file', state.file);

    var url;
    if (state.mode === 'detect') {
      fd.append('conf', els.confSlider.value);
      url = '/api/detect';
    } else if (state.mode === 'segment') {
      fd.append('text', els.promptInput.value.trim());
      fd.append('threshold', els.segThresholdSlider.value);
      url = '/api/segment';
    } else {
      fd.append('category', els.categorySelect.value);
      fd.append('threshold', els.threshSlider.value);
      url = '/api/inspect';
    }

    fetch(url, { method: 'POST', body: fd })
      .then(function (r) { return r.json(); })
      .then(function (json) {
        if (!json.ok) {
          var err = json.error || { code: 'UNKNOWN', message: '未知错误' };
          if (err.code === 'MODEL_NOT_READY') {
            setStatus('模型未就绪: ' + err.message, 'error');
            refreshModels();
          } else {
            setStatus(err.code + ': ' + err.message, 'error');
          }
          return;
        }
        state.lastResult = json;
        if (state.mode === 'detect') {
          renderDetect(json);
        } else if (state.mode === 'segment') {
          renderSegment(json, false);
        } else {
          renderInspect(json);
        }
        setStatus('完成, 耗时 ' + resultMs(json) + ' ms', 'ok');
      })
      .catch(function (err) {
        setStatus('请求失败: ' + err.message, 'error');
      })
      .finally(function () {
        els.runBtn.disabled = false;
      });
  }

  // 目标检测: 原图 + 检测框
  function renderDetect(json) {
    loadImage(state.dataURL).then(function (img) {
      var scale = drawImageFit(img);
      json.detections.forEach(function (det) {
        var x1 = det.box[0] * scale, y1 = det.box[1] * scale;
        var x2 = det.box[2] * scale, y2 = det.box[3] * scale;
        var w = x2 - x1, h = y2 - y1;

        ctx.strokeStyle = '#f5a623';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, w, h);

        var text = det.label + ' ' + Math.round(det.confidence * 100) + '%';
        ctx.font = 'bold 14px Consolas, monospace';
        var tw = ctx.measureText(text).width;
        ctx.fillStyle = '#f5a623';
        ctx.fillRect(x1, Math.max(0, y1 - 20), tw + 10, 20);
        ctx.fillStyle = '#12151c';
        ctx.fillText(text, x1 + 5, Math.max(14, y1 - 6));
      });

      var n = json.detections.length;
      if (n > 0) {
        setVerdict('检出 ' + n + ' 个目标', 'info');
      } else {
        setVerdict('未检出目标', 'ok');
      }
      els.metaBox.textContent = '耗时 ' + json.inference_ms + ' ms · 置信度阈值 ' + els.confSlider.value;
      pushHistory({
        mode: 'detect',
        file: state.file ? state.file.name : '',
        verdict: n > 0 ? 'info' : 'ok',
        verdictText: n > 0 ? '检出 ' + n + ' 个目标' : '未检出目标',
        thumb: makeThumb(),
        detail: json.detections.map(function (d) {
          return d.label + ' ' + Math.round(d.confidence * 100) + '%';
        }).join(', '),
      });
    });
  }

  // 瑕疵检测: 原图 + 热力图/叠加图
  function renderInspect(json) {
    els.viewRow.classList.remove('hidden');
    var view = document.querySelector('input[name="view"]:checked').value;

    var src = state.dataURL;
    if (view === 'overlay') { src = 'data:image/png;base64,' + json.overlay_png_base64; }
    if (view === 'heatmap') { src = 'data:image/png;base64,' + json.heatmap_png_base64; }

    loadImage(src).then(function (img) {
      drawImageFit(img);
      if (json.is_ng) {
        setVerdict('NG', 'ng');
      } else {
        setVerdict('OK', 'ok');
      }
      els.metaBox.textContent =
        '类别 ' + json.category +
        ' · 异常分数 ' + json.score +
        ' · 阈值 ' + json.threshold +
        ' · 耗时 ' + json.inference_ms + ' ms';
      pushHistory({
        mode: 'inspect',
        file: state.file ? state.file.name : '',
        verdict: json.is_ng ? 'ng' : 'ok',
        verdictText: (json.is_ng ? 'NG' : 'OK') + ' · ' + json.category + ' · 分数 ' + json.score,
        thumb: makeThumb(),
        detail: '阈值 ' + json.threshold + ' · ' + json.inference_ms + ' ms',
      });
    });
  }

  // 文本分割: mask / overlay 叠加图 + 实例清单
  function renderMaskChips(instances) {
    els.maskRow.innerHTML = '';

    var label = document.createElement('span');
    label.className = 'mask-label';
    label.textContent = '实例';
    els.maskRow.appendChild(label);

    if (!instances || instances.length === 0) {
      var none = document.createElement('span');
      none.className = 'mask-empty';
      none.textContent = '未检出实例';
      els.maskRow.appendChild(none);
      return;
    }

    instances.forEach(function (it, i) {
      var chip = document.createElement('span');
      chip.className = 'mask-chip';
      chip.title =
        '实例 #' + (i + 1) +
        ' · 分数 ' + Number(it.score).toFixed(3) +
        (it.box ? ' · box ' + it.box.join(', ') : '') +
        (it.area_px != null ? ' · 面积 ' + it.area_px + ' px' : '') +
        (it.area_ratio != null ? ' (' + Math.round(it.area_ratio * 100) + '%)' : '');

      var idx = document.createElement('span');
      idx.className = 'idx';
      idx.textContent = '#' + (i + 1);

      var name = document.createElement('span');
      name.className = 'name';
      name.textContent = it.concept || '-';   // 服务端字符串走 textContent

      var score = document.createElement('span');
      score.className = 'score';
      score.textContent = Number(it.score).toFixed(3);

      chip.appendChild(idx);
      chip.appendChild(name);
      chip.appendChild(score);

      if (it.area_ratio != null) {
        var area = document.createElement('span');
        area.className = 'area';
        area.textContent = Math.round(it.area_ratio * 100) + '%';
        chip.appendChild(area);
      }
      els.maskRow.appendChild(chip);
    });
  }

  function renderSegment(json, viewOnly) {
    var result = json.result || {};
    var instances = result.instances || [];
    var concepts = result.concepts || [];
    var best = 0;
    instances.forEach(function (it) { if (it.score > best) { best = it.score; } });

    els.viewRow.classList.remove('hidden');
    els.maskRow.classList.remove('hidden');
    renderMaskChips(instances);

    var b64 = checkedView() === 'mask' ? result.mask_png_base64 : result.overlay_png_base64;
    if (!b64) { b64 = result.overlay_png_base64 || result.mask_png_base64; }
    if (!b64) {
      setStatus('返回结果缺少 mask/overlay 图像', 'error');
      return;
    }

    loadImage('data:image/png;base64,' + b64).then(function (img) {
      drawImageFit(img);
      els.emptyHint.style.display = 'none';

      var n = instances.length;
      setVerdict(n + ' 个实例', 'info');
      els.metaBox.textContent =
        '概念: ' + (concepts.length ? concepts.join(',') : '-') +
        ' · 最高分 ' + (n ? best.toFixed(3) : '-') +
        ' · 耗时 ' + result.inference_ms + ' ms';

      if (viewOnly) { return; }   // 切视图重绘不重复写历史
      pushHistory({
        mode: 'segment',
        file: state.file ? state.file.name : '',
        verdict: 'info',
        verdictText: n + ' 个实例',
        thumb: makeThumb(),
        detail: '概念 ' + (concepts.length ? concepts.join(',') : '-') +
          ' · 最高分 ' + (n ? best.toFixed(3) : '-') + ' · ' + result.inference_ms + ' ms',
      });
    }).catch(function () {
      setStatus('分割结果图像解码失败', 'error');
    });
  }

  // ------------------------------------------------------------------
  // 历史记录(localStorage)
  // ------------------------------------------------------------------
  function readHistory() {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    } catch (e) {
      return [];
    }
  }

  function pushHistory(item) {
    var list = readHistory();
    list.unshift({
      time: new Date().toLocaleString(),
      mode: item.mode,
      file: item.file,
      verdict: item.verdict,
      verdictText: item.verdictText,
      thumb: item.thumb,
      detail: item.detail,
    });
    if (list.length > 50) { list = list.slice(0, 50); }
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
    } catch (e) {
      // 存储超限时丢弃最旧的一半再试一次
      list = list.slice(0, 20);
      try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); } catch (e2) {}
    }
    renderHistory();
  }

  function renderHistory() {
    var list = readHistory();
    els.historyList.innerHTML = '';
    if (list.length === 0) {
      els.historyList.innerHTML = '<div class="history-empty">暂无记录</div>';
      return;
    }
    list.forEach(function (item) {
      var div = document.createElement('div');
      div.className = 'history-item';
      div.innerHTML =
        '<img src="' + item.thumb + '" alt="thumb" />' +
        '<div class="h-meta">' +
        '<div class="h-verdict ' + item.verdict + '">' + item.verdictText + '</div>' +
        '<div>' + item.time + '</div>' +
        '<div>' + item.file + '</div>' +
        '</div>';
      div.title = item.detail || '';
      div.addEventListener('click', function () {
        loadImage(item.thumb).then(function (img) {
          els.emptyHint.style.display = 'none';
          drawImageFit(img);
          setVerdict(item.verdictText.split(' ')[0], item.verdict);
          els.metaBox.textContent = item.time + ' · ' + (item.detail || '');
        });
      });
      els.historyList.appendChild(div);
    });
  }

  // ------------------------------------------------------------------
  // 事件绑定
  // ------------------------------------------------------------------
  els.dropzone.addEventListener('click', function () { els.fileInput.click(); });
  els.dropzone.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { els.fileInput.click(); }
  });
  els.fileInput.addEventListener('change', function (e) {
    handleFile(e.target.files[0]);
    els.fileInput.value = '';
  });
  els.dropzone.addEventListener('dragover', function (e) {
    e.preventDefault();
    els.dropzone.classList.add('dragover');
  });
  els.dropzone.addEventListener('dragleave', function () {
    els.dropzone.classList.remove('dragover');
  });
  els.dropzone.addEventListener('drop', function (e) {
    e.preventDefault();
    els.dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) { handleFile(e.dataTransfer.files[0]); }
  });

  document.querySelectorAll('input[name="mode"]').forEach(function (radio) {
    radio.addEventListener('change', function (e) { setMode(e.target.value); });
  });
  document.querySelectorAll('input[name="view"]').forEach(function (radio) {
    radio.addEventListener('change', function () {
      if (!state.lastResult) { return; }
      if (state.mode === 'inspect') { renderInspect(state.lastResult); }
      else if (state.mode === 'segment') { renderSegment(state.lastResult, true); }
    });
  });
  els.categorySelect.addEventListener('change', updateRunButton);

  els.confSlider.addEventListener('input', function () {
    els.confVal.textContent = Number(els.confSlider.value).toFixed(2);
  });
  els.threshSlider.addEventListener('input', function () {
    els.threshVal.textContent = Number(els.threshSlider.value).toFixed(2);
  });
  els.segThresholdSlider.addEventListener('input', function () {
    els.segThresholdVal.textContent = Number(els.segThresholdSlider.value).toFixed(2);
  });
  els.promptInput.addEventListener('input', updateRunButton);

  els.runBtn.addEventListener('click', run);
  els.clearHistory.addEventListener('click', function () {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });

  // ------------------------------------------------------------------
  // 启动
  // ------------------------------------------------------------------
  setMode('detect');
  renderHistory();
  refreshModels();
})();
