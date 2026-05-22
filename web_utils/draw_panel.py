from IPython.display import HTML, Image, display
from google.colab.output import eval_js
from base64 import b64decode
import tempfile


from IPython.display import HTML, Image, display
from google.colab.output import eval_js
from base64 import b64decode
import tempfile


canvas_html = """
<style>
  canvas {
    border:2px dashed black;
    cursor: crosshair;
  }
  .btn {
    display: inline-block;
    margin: 6px 4px;
    padding: 6px 14px;
    background-color: #1fa3ec;
    border: 0;
    border-radius:0.317rem;
    color:white;
    height:32px;
    font-size:0.85rem;
    cursor:pointer;
  }
  .btn:hover:enabled {
    opacity: 0.75;
  }
  .btn:disabled {
    background-color: gray;
  }
  .btn-clear {
    background-color: #e05a3a;
  }
  .btn-row {
    display: flex;
    justify-content: center;
    gap: 8px;
  }
  #preview-display {
    border:2px dashed gray;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
  }
  .preview-label {
    color: gray;
    text-align: center;
    margin-top: 4px;
    font-size: 0.75rem;
  }
</style>
<div style='display:inline-flex; gap:20px; align-items:flex-start'>
  <div style='display:inline-grid'>
    <canvas width=%d height=%d></canvas>
    <div class="btn-row">
      <button class="btn" id="btn-done">Done ✓</button>
      <button class="btn btn-clear" id="btn-clear">Clear ✗</button>
    </div>
  </div>
  %s
</div>
<script>
var canvas = document.querySelector('canvas')
var ctx = canvas.getContext('2d')
ctx.lineWidth = %d
ctx.lineCap = 'round'
ctx.lineJoin = 'round'
var btnDone = document.getElementById('btn-done')
var btnClear = document.getElementById('btn-clear')

var offscreen = document.createElement('canvas')
offscreen.width = %d
offscreen.height = %d
var offctx = offscreen.getContext('2d')

var displayCanvas = document.getElementById('preview-display')
var showPreview = %s
var blurRadius = %.2f

var mouse = {x: 0, y: 0, px: 0, py: 0}
var Touch = {x: 0, y: 0, px: 0, py: 0}

function updatePreview() {
  if (!showPreview || !displayCanvas) return

  var blurred = document.createElement('canvas')
  blurred.width = canvas.width
  blurred.height = canvas.height
  var bctx = blurred.getContext('2d')
  bctx.filter = 'blur(' + blurRadius + 'px)'
  bctx.drawImage(canvas, 0, 0)

  offctx.clearRect(0, 0, offscreen.width, offscreen.height)
  offctx.drawImage(blurred, 0, 0, offscreen.width, offscreen.height)

  var imageData = offctx.getImageData(0, 0, offscreen.width, offscreen.height)
  var d = imageData.data
  for (var i = 0; i < d.length; i += 4) {
    var alpha = d[i + 3]
    var v = 255 - alpha
    d[i] = d[i+1] = d[i+2] = v
    d[i + 3] = 255
  }
  offctx.putImageData(imageData, 0, 0)

  var dctx = displayCanvas.getContext('2d')
  dctx.imageSmoothingEnabled = false
  dctx.fillStyle = 'white'
  dctx.fillRect(0, 0, displayCanvas.width, displayCanvas.height)
  dctx.drawImage(offscreen, 0, 0, displayCanvas.width, displayCanvas.height)
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (showPreview && displayCanvas) {
    var dctx = displayCanvas.getContext('2d')
    dctx.fillStyle = 'white'
    dctx.fillRect(0, 0, displayCanvas.width, displayCanvas.height)
  }
}

canvas.addEventListener('mousemove', function(e) {
  mouse.px = mouse.x
  mouse.py = mouse.y
  mouse.x = e.pageX - this.offsetLeft
  mouse.y = e.pageY - this.offsetTop
})
canvas.addEventListener('touchmove', function(e) {
  Touch.px = Touch.x
  Touch.py = Touch.y
  Touch.x = e.pageX - this.offsetLeft
  Touch.y = e.pageY - this.offsetTop
})

canvas.onmousedown = ()=>{
  ctx.beginPath()
  ctx.moveTo(mouse.x, mouse.y)
  mouse.px = mouse.x
  mouse.py = mouse.y
  canvas.addEventListener('mousemove', onPaint)
}
canvas.ontouchstart = ()=>{
  ctx.beginPath()
  ctx.moveTo(Touch.x, Touch.y)
  Touch.px = Touch.x
  Touch.py = Touch.y
  canvas.addEventListener('touchmove', onPaint2)
}
canvas.onmouseup = ()=>{
  canvas.removeEventListener('mousemove', onPaint)
}
canvas.ontouchend = ()=>{
  canvas.removeEventListener('touchmove', onPaint2)
}

var onPaint = ()=>{
  var midX = (mouse.px + mouse.x) / 2
  var midY = (mouse.py + mouse.y) / 2
  ctx.quadraticCurveTo(mouse.px, mouse.py, midX, midY)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(midX, midY)
  updatePreview()
}
var onPaint2 = ()=>{
  var midX = (Touch.px + Touch.x) / 2
  var midY = (Touch.py + Touch.y) / 2
  ctx.quadraticCurveTo(Touch.px, Touch.py, midX, midY)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(midX, midY)
  updatePreview()
}

btnClear.onclick = ()=>{
  clearCanvas()
}

var data = new Promise(resolve=>{
  btnDone.onclick = ()=>{
    resolve(canvas.toDataURL('image/png'))
    canvas.onmousedown = ()=>{}
    btnDone.style.visibility = 'hidden'
    btnClear.style.visibility = 'hidden'
  }
})
</script>
"""

from PIL import Image, ImageOps

class DrawPanel(object):

  def draw(self, size=(100,100), line_width=3, scale=1.0, show_preview=True, blur=None):
    w, h = size[0], size[1]
    line_width_scaled = line_width * scale
    blur_radius = blur if blur is not None else scale / 3

    preview_display_w = max(int(w * scale), 100)
    preview_display_h = max(int(h * scale), 100)

    if show_preview:
      preview_html = PREVIEW_HTML % (preview_display_w, preview_display_h, w, h)
    else:
      preview_html = ''

    display(HTML(canvas_html % (
      int(w * scale), int(h * scale),
      preview_html,
      line_width_scaled,
      w, h,
      'true' if show_preview else 'false',
      blur_radius
    )))

    data = eval_js("data")
    binary = b64decode(data.split(',')[1])

    from io import BytesIO
    buffer = BytesIO()
    buffer.write(binary)

    image = Image.open(buffer)
    image = image.resize((w, h))
    gray_image = ImageOps.grayscale(image)
    image.show()
    gray_image.show()
    return image

  def draw_to_file(self, filename='drawing.png', w=200, h=200, line_width=3):
    display(HTML(canvas_html % (w, h, '', line_width, w, h, 'false', 0.0)))
    data = eval_js("data")
    binary = b64decode(data.split(',')[1])
    filename = tempfile.mkdtemp() + '/' + filename
    with open(filename, 'wb') as f:
      f.write(binary)

    return filename

PREVIEW_HTML = """
  <div style='display:inline-grid; margin-top:2px'>
    <canvas id="preview-display" width=%d height=%d></canvas>
    <div class="preview-label">Scaled Preview (%dx%d)</div>
  </div>
"""


from PIL import Image, ImageOps

class DrawPanel(object):

  def draw(self, size=(100,100), line_width=3, scale=1.0, show_preview=True, blur=None):
    w, h = size[0], size[1]
    line_width_scaled = line_width * scale
    blur_radius = blur if blur is not None else scale / 3

    preview_display_w = max(int(w * scale), 100)
    preview_display_h = max(int(h * scale), 100)

    if show_preview:
      preview_html = PREVIEW_HTML % (preview_display_w, preview_display_h, w, h)
    else:
      preview_html = ''

    display(HTML(canvas_html % (
      int(w * scale), int(h * scale),
      preview_html,
      line_width_scaled,
      w, h,
      'true' if show_preview else 'false',
      blur_radius
    )))

    data = eval_js("data")
    binary = b64decode(data.split(',')[1])

    from io import BytesIO
    buffer = BytesIO()
    buffer.write(binary)

    image = Image.open(buffer)
    image = image.resize((w, h))
    gray_image = ImageOps.grayscale(image)
    image.show()
    gray_image.show()
    return image

  def draw_to_file(self, filename='drawing.png', w=200, h=200, line_width=3):
    display(HTML(canvas_html % (w, h, '', line_width, w, h, 'false', 0.0)))
    data = eval_js("data")
    binary = b64decode(data.split(',')[1])
    filename = tempfile.mkdtemp() + '/' + filename
    with open(filename, 'wb') as f:
      f.write(binary)

    return filename