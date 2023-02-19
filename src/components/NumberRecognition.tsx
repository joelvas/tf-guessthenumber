import * as tf from '@tensorflow/tfjs';
import { useEffect, useState } from 'react'
import { fabric } from "fabric";

const NumberRecognition = () => {
  const [modelo, setModelo] = useState<tf.LayersModel | null>(null)
  const [canvas, setCanvas] = useState<fabric.Canvas | null>(null)
  const [result, setResult] = useState<number | null>(null)

  const loadModel = async () => {
    console.log("Loading model")
    const result = await tf.loadLayersModel('./tf-models/model.json')
    console.log("Loaded!")
    result.predict(tf.zeros([1, 28, 28, 1]))
    setModelo(result)
  }

  const loadCanvas = () => {
    const newCanvas = new fabric.Canvas(document.querySelector('#canvasId') as HTMLCanvasElement)
    newCanvas.isDrawingMode = true;
    newCanvas.freeDrawingBrush.width = 5;
    newCanvas.freeDrawingBrush.color = '#4b4b4b';
    newCanvas.on('path:created', (e: any) => {
      console.log({ canvas })
      e.path.set();
      newCanvas?.renderAll();
      setCanvas(newCanvas)
    });
    setCanvas(newCanvas)
  }

  const predict = async () => {
    console.log({ canvas })
    const image = tf.browser.fromPixels(canvas?.getElement() as HTMLCanvasElement)
    const resizedImage = tf.image.resizeBilinear(image, [28, 28]).sum(2).expandDims(0).expandDims(-1)
    const res = modelo?.predict(resizedImage) as tf.Tensor<tf.Rank>;
    const resArr = res.dataSync()
    const mayorIndice = resArr.indexOf(Math.max.apply(null, resArr as any))

    setResult(mayorIndice)
  }

  function limpiar() {
    console.log({ canvas })
    canvas?.clear()
    setResult(null)
  }

  useEffect(() => {
    loadModel()
    loadCanvas()
  }, [])

  return (
    <div>
      {
        modelo ? <span>Loaded!</span> : <span>Loading...</span>
      }
      <canvas style={{ border: '1px solid grey' }} id="canvasId" width="150" height="150"></canvas>
      <br />
      <p>{result}</p>
      <button onClick={limpiar}>Limpiar</button>
      <button onClick={predict}>Predecir</button>
    </div>
  )
}
export default NumberRecognition
