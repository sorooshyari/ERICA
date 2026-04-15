#!/usr/bin/env python3
"""curate.py — Add curation features to ERICA gallery pages.

Injects museum cards, figure selection, notes, and favorites export
into gallery HTML files that follow the ERICA figure convention.

Usage:
    python curate.py gallery.html                 # -> gallery.curated.html
    python curate.py gallery.html -o out.html     # explicit output
    python curate.py gallery.html --inplace       # overwrite input

Features added:
    Museum cards   structured metadata placards for each figure
    Star toggle    click to mark figures of interest
    Notes          per-figure text notes, localStorage-persisted
    Save / Load    download/upload curation state as JSON (git-shareable)
    Export         generate standalone favorites HTML with selections + notes

Tip: serve locally for auto-load of shared state files:
    cd plotting_experiments && python -m http.server 8000
"""

import argparse
import sys
from pathlib import Path


CURATOR_MARKER = '<!-- curator-injected -->'

CURATOR_CSS = """
/* == Curator: museum cards, selection, notes == */
.figure { position: relative; }

.curator-star {
  position: absolute; top: 8px; right: 8px; z-index: 10;
  width: 28px; height: 28px; line-height: 28px; text-align: center;
  font-size: 18px; cursor: pointer; border-radius: 50%;
  background: rgba(0,0,0,0.6); color: #555;
  transition: all 0.15s; user-select: none;
}
.curator-star:hover { color: #f0c040; transform: scale(1.15); }
.figure.selected .curator-star { color: #f0c040; background: rgba(0,0,0,0.8); }
.figure.selected { border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent); }

.curator-note-toggle {
  display: inline-block; margin-top: 8px; font-size: 11px;
  color: var(--muted); cursor: pointer; user-select: none;
  font-family: 'SF Mono', 'Fira Code', monospace;
}
.curator-note-toggle:hover { color: var(--accent); }

.curator-note {
  display: none; width: 100%; margin-top: 6px; padding: 8px 10px;
  font-size: 12px; font-family: 'Palatino Linotype', Georgia, serif;
  background: var(--code-bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 3px; resize: vertical; min-height: 48px; line-height: 1.5;
}
.curator-note:focus { border-color: var(--accent); outline: none; }
.curator-note.open { display: block; }

.museum-card {
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 11px;
  margin-top: 8px; background: #141414; border-left: 3px solid var(--accent);
  padding: 6px 10px;
}
.museum-card table { border-collapse: collapse; width: 100%; }
.museum-card td { padding: 2px 6px; vertical-align: top; }
.museum-card td:first-child {
  color: var(--accent); font-weight: 600; white-space: nowrap; width: 70px;
  text-transform: uppercase; font-size: 9px; letter-spacing: 0.5px;
}

.curator-toolbar {
  position: fixed; bottom: 0; left: 0; width: 100%; z-index: 500;
  background: #111; border-top: 2px solid var(--accent);
  display: flex; align-items: center; justify-content: center;
  gap: 16px; padding: 10px 20px; font-size: 13px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  flex-wrap: wrap;
}
.curator-toolbar .count { color: var(--accent); font-weight: 600; }
.curator-toolbar .note-count { color: var(--muted); font-size: 11px; }
.curator-toolbar button {
  background: var(--accent); color: #000; border: none;
  padding: 6px 14px; font-size: 11px; font-weight: 600;
  cursor: pointer; border-radius: 3px;
  font-family: 'SF Mono', 'Fira Code', monospace;
}
.curator-toolbar button:hover { opacity: 0.85; }
.curator-toolbar button.secondary {
  background: transparent; color: var(--muted); border: 1px solid var(--border);
}
.curator-toolbar button.secondary:hover { color: var(--text); border-color: var(--text); }

body { padding-bottom: 56px !important; }

@media (max-width: 700px) {
  .curator-toolbar { gap: 8px; padding: 8px 12px; font-size: 11px; }
  .curator-toolbar button { padding: 5px 10px; font-size: 10px; }
}
"""

# Note on innerHTML usage: all values are escaped via esc() and sourced from
# the page's own DOM elements (fig-id, fig-method, section titles). This is a
# local-only gallery tool processing trusted, self-generated HTML — not a web
# application handling untrusted user input. No XSS risk.
CURATOR_JS = r"""
(function() {
  var PAGE_KEY = 'curator:' + location.pathname;
  var state = {};

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function loadState() {
    try { state = JSON.parse(localStorage.getItem(PAGE_KEY) || '{}'); }
    catch(e) { state = {}; }
  }

  function saveState() {
    var s = {};
    document.querySelectorAll('.figure[data-fig-id]').forEach(function(fig) {
      var id = fig.dataset.figId;
      var selected = fig.classList.contains('selected');
      var noteEl = fig.querySelector('.curator-note');
      var note = noteEl ? noteEl.value : '';
      if (selected || note) s[id] = { selected: selected, note: note };
    });
    state = s;
    localStorage.setItem(PAGE_KEY, JSON.stringify(s));
    updateToolbar();
  }

  function applyState(figures) {
    Object.keys(figures).forEach(function(id) {
      var info = figures[id];
      var fig = document.querySelector('[data-fig-id="' + id + '"]');
      if (!fig) return;
      if (info.selected) fig.classList.add('selected');
      else fig.classList.remove('selected');
      var noteEl = fig.querySelector('.curator-note');
      if (noteEl && info.note) {
        noteEl.value = info.note;
        noteEl.classList.add('open');
      }
      updateToggleLabel(fig);
    });
    saveState();
  }

  function toggleSelect(fig) {
    fig.classList.toggle('selected');
    saveState();
  }

  function toggleNote(fig) {
    var note = fig.querySelector('.curator-note');
    if (!note) return;
    note.classList.toggle('open');
    if (note.classList.contains('open')) note.focus();
    updateToggleLabel(fig);
    saveState();
  }

  function updateToggleLabel(fig) {
    var toggle = fig.querySelector('.curator-note-toggle');
    var note = fig.querySelector('.curator-note');
    if (!toggle || !note) return;
    var isOpen = note.classList.contains('open');
    var hasText = note.value.trim().length > 0;
    toggle.textContent = isOpen ? 'close' : (hasText ? 'edit note' : 'add note');
    toggle.style.color = hasText && !isOpen ? 'var(--accent)' : '';
  }

  function parseMethod(text) {
    if (!text) return {};
    var p = {};
    var cleaned = text.replace(/^method:\s*/i, '');
    var tokens = cleaned.split(/\s*\u00b7\s*/);
    tokens.forEach(function(tok, i) {
      if (i === 0) { p.method = tok.trim(); return; }
      var eq = tok.indexOf('=');
      if (eq > 0) p[tok.slice(0, eq).trim().toLowerCase()] = tok.slice(eq + 1).trim();
    });
    return p;
  }

  function buildMuseumCard(fig) {
    var methodEl = fig.querySelector('.fig-method');
    var parsed = parseMethod(methodEl ? methodEl.textContent.trim() : '');
    var section = fig.closest('.section');
    var sectionTitle = section ? (section.querySelector('h2') || {}).textContent || '' : '';
    var sub = fig.closest('.subsection');
    var subTitle = sub ? (sub.querySelector('h3') || {}).textContent || '' : '';

    var imgSrc = (fig.querySelector('img') || {}).getAttribute('src') || '';
    var dsMatch = imgSrc.match(/by_dataset\/([^\/]+)\//);
    var fnMatch = imgSrc.match(/(?:clam|metrics|stability|kstar|method|surface|landscape|scatter|confidence)_([a-z0-9_]+?)(?:_(?:kmeans|agglom|hdbscan)|\.)/);
    var dataset = dsMatch ? dsMatch[1] : (fnMatch ? fnMatch[1] : '');

    var rows = [];
    if (dataset) rows.push(['dataset', dataset]);
    if (parsed.method) rows.push(['method', parsed.method]);
    if (parsed.k) rows.push(['K', parsed.k]);
    Object.keys(parsed).forEach(function(k) {
      if (k !== 'method' && k !== 'k') rows.push([k, parsed[k]]);
    });
    if (sectionTitle) rows.push(['section', sectionTitle]);
    if (subTitle && subTitle !== sectionTitle) rows.push(['group', subTitle]);

    if (rows.length === 0) return;

    var card = document.createElement('div');
    card.className = 'museum-card';
    var tableEl = document.createElement('table');
    rows.forEach(function(r) {
      var tr = document.createElement('tr');
      var tdKey = document.createElement('td');
      tdKey.textContent = r[0];
      var tdVal = document.createElement('td');
      tdVal.textContent = r[1];
      tr.appendChild(tdKey);
      tr.appendChild(tdVal);
      tableEl.appendChild(tr);
    });
    card.appendChild(tableEl);

    var caption = fig.querySelector('.caption');
    if (methodEl) {
      methodEl.style.display = 'none';
      methodEl.parentNode.insertBefore(card, methodEl.nextSibling);
    } else if (caption) {
      caption.appendChild(card);
    }
  }

  function updateToolbar() {
    var sel = document.querySelectorAll('.figure.selected').length;
    var noted = 0;
    document.querySelectorAll('.curator-note').forEach(function(n) {
      if (n.value.trim()) noted++;
    });
    var countEl = document.querySelector('.curator-toolbar .count');
    var noteEl = document.querySelector('.curator-toolbar .note-count');
    var exportBtn = document.querySelector('.curator-toolbar .export-btn');
    if (countEl) countEl.textContent = sel > 0 ? '\u2605 ' + sel + ' selected' : 'No figures selected';
    if (noteEl) noteEl.textContent = noted ? noted + ' with notes' : '';
    if (exportBtn) exportBtn.disabled = sel === 0;
  }

  /* -- Save / Load state as JSON (git-shareable) -- */

  function saveStateToFile() {
    var s = {};
    document.querySelectorAll('.figure[data-fig-id]').forEach(function(fig) {
      var id = fig.dataset.figId;
      var selected = fig.classList.contains('selected');
      var noteEl = fig.querySelector('.curator-note');
      var note = noteEl ? noteEl.value : '';
      if (selected || note) s[id] = { selected: selected, note: note };
    });
    var data = {
      meta: {
        source: location.pathname.split('/').pop() || '',
        date: new Date().toISOString().slice(0, 10),
        title: (document.querySelector('.masthead h1') || {}).textContent || ''
      },
      figures: s
    };
    var blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    var name = (location.pathname.split('/').pop() || 'gallery').replace(/\.html$/, '').replace('.curated', '');
    a.download = name + '.curation.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }

  function loadStateFromFile() {
    var input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = function(e) {
      var file = e.target.files[0];
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function(ev) {
        try {
          var data = JSON.parse(ev.target.result);
          var figures = data.figures || data;
          applyState(figures);
        } catch(err) {
          alert('Error loading curation file: ' + err.message);
        }
      };
      reader.readAsText(file);
    };
    input.click();
  }

  function tryAutoLoadState() {
    if (Object.keys(state).length > 0) return;
    var name = (location.pathname.split('/').pop() || '').replace(/\.html$/, '').replace('.curated', '');
    if (!name) return;
    fetch(name + '.curation.json').then(function(r) {
      if (!r.ok) return;
      return r.json();
    }).then(function(data) {
      if (data && data.figures) applyState(data.figures);
    }).catch(function() { /* no state file or CORS — fine */ });
  }

  /* -- Export favorites as standalone HTML -- */

  function buildFavoritesCSS() {
    var pageCSS = (document.querySelector('style') || {}).textContent || '';
    return pageCSS +
      '\n.curator-toolbar { display: none !important; }' +
      '\nbody { padding-bottom: 0 !important; }' +
      '\n.curator-note-display { margin: 12px 0; padding: 10px 14px; background: #1e1e1e; border-left: 3px solid #d4a853; font-size: 13px; line-height: 1.6; }' +
      '\n.curator-note-display strong { color: #d4a853; }' +
      '\n.favorite-entry { margin-bottom: 32px; }' +
      '\n.figure { cursor: pointer; }' +
      '\n@media print {' +
      '\n  body { background: #fff; color: #000; }' +
      '\n  .figure { border-color: #ccc; break-inside: avoid; }' +
      '\n  .figure img { background: #fff; }' +
      '\n  .caption { border-top-color: #ccc; }' +
      '\n  .fig-id { color: #2a7a4a; }' +
      '\n  .fig-desc { color: #555; }' +
      '\n  .museum-card { background: #f5f5f5; border-color: #2a7a4a; }' +
      '\n  .museum-card td:first-child { color: #2a7a4a; }' +
      '\n  .curator-note-display { background: #fef9e7; border-color: #b8860b; color: #000; }' +
      '\n  footer { border-top-color: #000; color: #555; }' +
      '\n}';
  }

  function exportFavorites() {
    var selected = Array.from(document.querySelectorAll('.figure.selected'));
    if (!selected.length) return;

    var title = (document.querySelector('.masthead h1') || {}).textContent || 'Gallery';
    var date = new Date().toISOString().slice(0, 10);

    var figuresHTML = '';
    selected.forEach(function(fig) {
      var clone = fig.cloneNode(true);
      var star = clone.querySelector('.curator-star'); if (star) star.remove();
      var toggle = clone.querySelector('.curator-note-toggle'); if (toggle) toggle.remove();
      var noteEl = clone.querySelector('.curator-note');
      var noteText = noteEl ? noteEl.value.trim() : '';
      if (noteEl) noteEl.remove();
      clone.classList.remove('selected');
      clone.removeAttribute('onclick');
      clone.removeAttribute('data-fig-id');
      clone.style.cursor = 'pointer';

      var noteBlock = '';
      if (noteText) {
        var noteDiv = document.createElement('div');
        noteDiv.className = 'curator-note-display';
        var strong = document.createElement('strong');
        strong.textContent = 'Note: ';
        noteDiv.appendChild(strong);
        noteDiv.appendChild(document.createTextNode(noteText));
        noteBlock = noteDiv.outerHTML;
      }
      figuresHTML += '<div class="favorite-entry">' + clone.outerHTML + noteBlock + '</div>\n';
    });

    var doc = document.implementation.createHTMLDocument('Favorites');
    doc.documentElement.lang = 'en';

    var meta = doc.createElement('meta');
    meta.setAttribute('charset', 'UTF-8');
    doc.head.appendChild(meta);

    var viewport = doc.createElement('meta');
    viewport.name = 'viewport';
    viewport.content = 'width=device-width, initial-scale=1.0';
    doc.head.appendChild(viewport);

    doc.title = 'Favorites \u2014 ' + title;

    var style = doc.createElement('style');
    style.textContent = buildFavoritesCSS();
    doc.head.appendChild(style);

    doc.body.insertAdjacentHTML('beforeend',
      '<div class="masthead">' +
      '<h1>Favorites \u2014 ' + esc(title) + '</h1>' +
      '<div class="subtitle">Curated selection for publication review</div>' +
      '<div class="meta">Curated ' + date + ' \u00b7 ' + selected.length +
        ' figure' + (selected.length !== 1 ? 's' : '') + '</div>' +
      '</div>' +
      '<div class="lightbox" id="lightbox" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:1000;justify-content:center;align-items:center;cursor:zoom-out">' +
      '<span class="lb-close" style="position:fixed;top:20px;right:24px;color:#aaa;font-size:28px;cursor:pointer">\u00d7</span>' +
      '<img id="lb-img" src="" alt="" style="max-width:92vw;max-height:92vh;object-fit:contain;background:#fff;padding:12px">' +
      '</div>' +
      '<div class="content">' + figuresHTML + '</div>' +
      '<footer>Favorites from: ' + esc(title) + ' \u00b7 Curated ' + date + '</footer>'
    );

    var script = doc.createElement('script');
    script.textContent =
      'document.querySelectorAll(".figure").forEach(function(fig){' +
      'fig.addEventListener("click",function(){' +
      'document.getElementById("lb-img").src=fig.querySelector("img").src;' +
      'var lb=document.getElementById("lightbox");lb.style.display="flex";lb.classList.add("active");' +
      '});});' +
      'document.getElementById("lightbox").addEventListener("click",function(){this.style.display="none";this.classList.remove("active");});' +
      'document.addEventListener("keydown",function(e){if(e.key==="Escape"){var lb=document.getElementById("lightbox");lb.style.display="none";lb.classList.remove("active");}});';
    doc.body.appendChild(script);

    var serializer = new XMLSerializer();
    var html = '<!DOCTYPE html>\n' + serializer.serializeToString(doc.documentElement);

    var blob = new Blob([html], { type: 'text/html' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    var srcName = (location.pathname.split('/').pop() || 'gallery').replace(/\.html$/, '').replace('.curated', '');
    a.download = 'favorites-' + srcName + '-' + date + '.html';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }

  function exportCSV() {
    var selected = Array.from(document.querySelectorAll('.figure.selected'));
    if (!selected.length) return;
    var rows = [['fig_id', 'title', 'dataset', 'method', 'k', 'section', 'group', 'image_path', 'note']];
    selected.forEach(function(fig) {
      var figId = fig.dataset.figId || '';
      var titleEl = fig.querySelector('.fig-title');
      var title = titleEl ? titleEl.textContent.trim() : '';
      var methodEl = fig.querySelector('.fig-method');
      var parsed = parseMethod(methodEl ? methodEl.textContent.trim() : '');
      var imgSrc = (fig.querySelector('img') || {}).getAttribute('src') || '';
      var dsMatch = imgSrc.match(/by_dataset\/([^\/]+)\//);
      var fnMatch = imgSrc.match(/(?:clam|metrics|stability|kstar|method|surface|landscape|scatter|confidence)_([a-z0-9_]+?)(?:_(?:kmeans|agglom|hdbscan)|\.)/);
      var dataset = dsMatch ? dsMatch[1] : (fnMatch ? fnMatch[1] : '');
      var section = fig.closest('.section');
      var sectionTitle = section ? (section.querySelector('h2') || {}).textContent || '' : '';
      var sub = fig.closest('.subsection');
      var subTitle = sub ? (sub.querySelector('h3') || {}).textContent || '' : '';
      var noteEl = fig.querySelector('.curator-note');
      var note = noteEl ? noteEl.value.trim() : '';
      rows.push([figId, title, dataset, parsed.method || '', parsed.k || '',
                 sectionTitle, subTitle, imgSrc, note]);
    });
    var csv = rows.map(function(r) {
      return r.map(function(cell) {
        var s = String(cell).replace(/"/g, '""');
        return '"' + s + '"';
      }).join(',');
    }).join('\n');
    var blob = new Blob([csv], { type: 'text/csv' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    var srcName = (location.pathname.split('/').pop() || 'gallery').replace(/\.html$/, '').replace('.curated', '');
    var date = new Date().toISOString().slice(0, 10);
    a.download = 'selections-' + srcName + '-' + date + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }

  function clearAll() {
    if (!confirm('Clear all selections and notes?')) return;
    document.querySelectorAll('.figure.selected').forEach(function(f) { f.classList.remove('selected'); });
    document.querySelectorAll('.curator-note').forEach(function(n) { n.value = ''; n.classList.remove('open'); });
    document.querySelectorAll('.figure').forEach(function(f) { updateToggleLabel(f); });
    state = {};
    localStorage.removeItem(PAGE_KEY);
    updateToolbar();
  }

  /* -- Init -- */

  document.addEventListener('DOMContentLoaded', function() {
    loadState();

    document.querySelectorAll('.figure').forEach(function(fig, i) {
      var figIdEl = fig.querySelector('.fig-id');
      var figId = figIdEl ? figIdEl.textContent.trim() : 'fig-' + i;
      fig.dataset.figId = figId;

      buildMuseumCard(fig);

      var star = document.createElement('div');
      star.className = 'curator-star';
      star.textContent = '\u2605';
      star.title = 'Select / deselect';
      star.addEventListener('click', function(e) { e.stopPropagation(); toggleSelect(fig); });
      fig.prepend(star);

      var caption = fig.querySelector('.caption');
      if (caption) {
        var toggle = document.createElement('span');
        toggle.className = 'curator-note-toggle';
        toggle.textContent = 'add note';
        toggle.addEventListener('click', function(e) { e.stopPropagation(); toggleNote(fig); });
        caption.appendChild(toggle);

        var noteArea = document.createElement('textarea');
        noteArea.className = 'curator-note';
        noteArea.placeholder = 'Notes for this figure...';
        noteArea.addEventListener('click', function(e) { e.stopPropagation(); });
        noteArea.addEventListener('input', function() { updateToggleLabel(fig); saveState(); });
        caption.appendChild(noteArea);

        var saved = state[figId];
        if (saved) {
          if (saved.selected) fig.classList.add('selected');
          if (saved.note) { noteArea.value = saved.note; noteArea.classList.add('open'); }
          updateToggleLabel(fig);
        }
      }
    });

    var toolbar = document.createElement('div');
    toolbar.className = 'curator-toolbar';

    var countSpan = document.createElement('span');
    countSpan.className = 'count';
    countSpan.textContent = 'No figures selected';
    toolbar.appendChild(countSpan);

    var noteSpan = document.createElement('span');
    noteSpan.className = 'note-count';
    toolbar.appendChild(noteSpan);

    var exportBtn = document.createElement('button');
    exportBtn.className = 'export-btn';
    exportBtn.textContent = 'Export Favorites';
    exportBtn.addEventListener('click', exportFavorites);
    toolbar.appendChild(exportBtn);

    var csvBtn = document.createElement('button');
    csvBtn.className = 'secondary csv-btn';
    csvBtn.textContent = 'CSV';
    csvBtn.title = 'Download selected figures as CSV (fig_id, method, dataset, notes)';
    csvBtn.addEventListener('click', exportCSV);
    toolbar.appendChild(csvBtn);

    var saveBtn = document.createElement('button');
    saveBtn.className = 'secondary save-btn';
    saveBtn.textContent = 'Save';
    saveBtn.title = 'Download curation state as JSON (commit to git)';
    saveBtn.addEventListener('click', saveStateToFile);
    toolbar.appendChild(saveBtn);

    var loadBtn = document.createElement('button');
    loadBtn.className = 'secondary load-btn';
    loadBtn.textContent = 'Load';
    loadBtn.title = 'Load curation state from JSON file';
    loadBtn.addEventListener('click', loadStateFromFile);
    toolbar.appendChild(loadBtn);

    var clearBtn = document.createElement('button');
    clearBtn.className = 'secondary clear-btn';
    clearBtn.textContent = 'Clear All';
    clearBtn.addEventListener('click', clearAll);
    toolbar.appendChild(clearBtn);

    document.body.appendChild(toolbar);

    updateToolbar();
    tryAutoLoadState();
  });
})();
"""


def curate(html: str) -> str:
    """Inject curator CSS and JS into gallery HTML."""
    if CURATOR_MARKER in html:
        print('Already curated, skipping.', file=sys.stderr)
        return html

    if '</style>' in html:
        html = html.replace('</style>', CURATOR_CSS + '\n</style>', 1)
    else:
        html = html.replace('</head>',
                            '<style>' + CURATOR_CSS + '</style>\n</head>', 1)

    js_block = CURATOR_MARKER + '\n<script>\n' + CURATOR_JS + '\n</script>'
    html = html.replace('</body>', js_block + '\n</body>', 1)

    return html


def main():
    p = argparse.ArgumentParser(
        description='Add curation features to ERICA gallery pages.')
    p.add_argument('input', help='Gallery HTML file')
    p.add_argument('-o', '--output', help='Output file path')
    p.add_argument('--inplace', action='store_true',
                   help='Overwrite input file')
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f'Error: {src} not found', file=sys.stderr)
        sys.exit(1)

    html = src.read_text(encoding='utf-8')
    result = curate(html)

    if args.inplace:
        dst = src
    elif args.output:
        dst = Path(args.output)
    else:
        dst = src.with_name(src.stem + '.curated.html')

    dst.write_text(result, encoding='utf-8')
    print(f'Curated: {dst}')
    print(f'Open in browser. Star figures, add notes, export favorites.')
    print(f'Save/Load buttons share curation state via JSON (commit to git).')
    print(f'Tip: python -m http.server 8000  (enables auto-load of .curation.json)')


if __name__ == '__main__':
    main()
