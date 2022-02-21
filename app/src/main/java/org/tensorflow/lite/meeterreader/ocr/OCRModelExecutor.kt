/*
 * Created by Renjith Sasidharan on 21/02/22, 11:57 PM
 * renjithks@gmail.com
 * Last modified 21/02/22, 11:57 PM
 * Copyright (c) 2022.
 * All rights reserved.
 */

package org.tensorflow.lite.meeterreader.ocr

import android.content.Context
import android.graphics.*
import android.util.Log
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils.bitmapToMat
import org.opencv.android.Utils.matToBitmap
import org.opencv.core.*
import org.opencv.core.Point
import org.opencv.dnn.Dnn.NMSBoxesRotated
import org.opencv.imgproc.Imgproc.boxPoints
import org.opencv.imgproc.Imgproc.getPerspectiveTransform
import org.opencv.imgproc.Imgproc.warpPerspective
import org.opencv.utils.Converters.vector_RotatedRect_to_Mat
import org.opencv.utils.Converters.vector_float_to_Mat
import org.tensorflow.lite.Interpreter
import java.lang.Math.abs


/**
 * Class to run the OCR models. The OCR process is broken down into 3 stages:
 * 1. Object detection
 *    using mobilenetV2 SSD for detecting display
 * 2. Text detection
 *    using [EAST model]
 * 3. Text recognition
 *    using [Keras OCR model]
 */
class OCRModelExecutor(context: Context) : AutoCloseable {

  private val recognitionResult: ByteBuffer
  private val displayInterpreter: Interpreter
  private val detectionInterpreter: Interpreter
  private val recognitionInterpreter: Interpreter

  private var ratioHeight = 0.toFloat()
  private var ratioWidth = 0.toFloat()
  private var indicesMat: MatOfInt
  private var displayBox: BoundingBox?
  private var boundingBoxesMat: MatOfRotatedRect
  private var detectedConfidencesMat: MatOfFloat
  private var ocrResults: String?

  init {
    try {
      if (!OpenCVLoader.initDebug()) throw Exception("Unable to load OpenCV")
      else Log.d(TAG, "OpenCV loaded")
    } catch (e: Exception) {
      val exceptionLog = "something went wrong: ${e.message}"
      Log.d(TAG, exceptionLog)
    }

    displayInterpreter = getInterpreter(context, displayDetectionModel)
    detectionInterpreter = getInterpreter(context, textDetectionModel)
    recognitionInterpreter = getInterpreter(context, textRecognitionModel)

    recognitionResult = ByteBuffer.allocateDirect(recognitionModelOutputSize * 8)
    recognitionResult.order(ByteOrder.nativeOrder())
    indicesMat = MatOfInt()
    boundingBoxesMat = MatOfRotatedRect()
    detectedConfidencesMat = MatOfFloat()
    ocrResults = null
    displayBox = null
  }

  /**
   * Get reading from an image
   * @param data
   *  - Bitmap representation of the image with 3 channels (width x height x channel).
   *  Needs a square image, with size >= 320 and multiple of 32. (Ex: 320x320x3, 512x512x3 etc)
   *  @return ModelExecutionResult
   */
  fun execute(data: Bitmap): ModelExecutionResult {
    if (data.height != data.width) {
      val exceptionLog = "Image should be square, got: (${data.height}, ${data.width}) pixels"
      Log.e(TAG, exceptionLog)

      return ModelExecutionResult(data, exceptionLog, null)
    }

    if (data.height < 320) {
      val exceptionLog = "Image size should be at least ${detectionImageHeight}, got: (${data.height} pixels"
      Log.e(TAG, exceptionLog)

      return ModelExecutionResult(data, exceptionLog, null)
    }

    if ((data.height % 32) != 0) {
      val exceptionLog = "Image size should be multiple of 32, got: (${data.height} pixels"
      Log.e(TAG, exceptionLog)

      return ModelExecutionResult(data, exceptionLog, null)
    }


    try {
      ratioHeight = data.height.toFloat() / detectionImageHeight
      ratioWidth = data.width.toFloat() / detectionImageWidth
      ocrResults = null

      detectDisplay(data)
      detectTexts(data)

      if (boundingBoxesMat.total() == 0L) {
        val exceptionLog = "No reading box detected"
        Log.e(TAG, exceptionLog)

        return ModelExecutionResult(data, exceptionLog, null)
      }

      val bitmapWithBoundingBoxes = recognizeTexts(data, displayBox, boundingBoxesMat, indicesMat)

      return ModelExecutionResult(bitmapWithBoundingBoxes, "OCR result", ocrResults)
    } catch (e: Exception) {
      val exceptionLog = "something went wrong: ${e.message}"
      Log.e(TAG, exceptionLog)

      val emptyBitmap = ImageUtils.createEmptyBitmap(displayImageSize, displayImageSize)
      return ModelExecutionResult(emptyBitmap, exceptionLog, null)
    }
  }

  private fun detectDisplay(data: Bitmap) {
    val detectionTensorImage =
      ImageUtils.bitmapToTensorImageForDisplayDetection(
        data,
        detectionImageWidth,
        detectionImageHeight,
        displayDetectionImageMean,
        displayDetectionImageStd
      )

    val detectionInputs = arrayOf(detectionTensorImage.buffer.rewind())
    val detectionOutputs: HashMap<Int, Any> = HashMap<Int, Any>()

    val boxes = Array(1) {
      Array(10) {
        FloatArray(4)
      }
    }
    val classes = Array(1) {
      FloatArray(
        10
      )
    }
    val scores = Array(1) {
      FloatArray(
        10
      )
    }
    val numOf = FloatArray(1)

    detectionOutputs.put(0, scores);
    detectionOutputs.put(1, boxes);
    detectionOutputs.put(2, numOf);
    detectionOutputs.put(3, classes);

    displayInterpreter.runForMultipleInputsOutputs(detectionInputs, detectionOutputs)
    val idx = scores[0].withIndex().maxByOrNull { it.value }?.index
    val box = if (idx != null) boxes[0][idx.toInt()] else null

    if (box != null) {
      //convert [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
      val bbox = BoundingBox()
      bbox.x_min = minOf(maxOf(box[1], 0.0f), 1.0f) * detectionImageWidth
      bbox.y_min = minOf(maxOf(box[0], 0.0f), 1.0f) * detectionImageHeight
      bbox.x_max = minOf(maxOf(box[3], 0.0f), 1.0f) * detectionImageWidth
      bbox.y_max = minOf(maxOf(box[2], 0.0f), 1.0f) * detectionImageHeight
      displayBox = bbox
    }
  }

  private fun detectTexts(data: Bitmap) {
    val detectionTensorImage =
      ImageUtils.bitmapToTensorImageForDetection(
        data,
        detectionImageWidth,
        detectionImageHeight
      )

    val detectionInputs = arrayOf(detectionTensorImage.buffer.rewind())
    val detectionOutputs: HashMap<Int, Any> = HashMap<Int, Any>()

    val detectionScores =
      Array(1) { Array(detectionOutputNumRows) { Array(detectionOutputNumCols) { FloatArray(1) } } }
    val detectionGeometries =
      Array(1) { Array(detectionOutputNumRows) { Array(detectionOutputNumCols) { FloatArray(5) } } }
    detectionOutputs.put(0, detectionScores)
    detectionOutputs.put(1, detectionGeometries)

    detectionInterpreter.runForMultipleInputsOutputs(detectionInputs, detectionOutputs)

    val transposeddetectionScores =
      Array(1) { Array(1) { Array(detectionOutputNumRows) { FloatArray(detectionOutputNumCols) } } }
    val transposedDetectionGeometries =
      Array(1) { Array(5) { Array(detectionOutputNumRows) { FloatArray(detectionOutputNumCols) } } }

    // transpose detection output tensors
    for (i in 0 until transposeddetectionScores[0][0].size) {
      for (j in 0 until transposeddetectionScores[0][0][0].size) {
        for (k in 0 until 1) {
          transposeddetectionScores[0][k][i][j] = detectionScores[0][i][j][k]
        }
        for (k in 0 until 5) {
          transposedDetectionGeometries[0][k][i][j] = detectionGeometries[0][i][j][k]
        }
      }
    }

    val detectedRotatedRects = ArrayList<RotatedRect>()
    val detectedConfidences = ArrayList<Float>()

    for (y in 0 until transposeddetectionScores[0][0].size) {
      val detectionScoreData = transposeddetectionScores[0][0][y]
      val detectionGeometryX0Data = transposedDetectionGeometries[0][0][y]
      val detectionGeometryX1Data = transposedDetectionGeometries[0][1][y]
      val detectionGeometryX2Data = transposedDetectionGeometries[0][2][y]
      val detectionGeometryX3Data = transposedDetectionGeometries[0][3][y]
      val detectionRotationAngleData = transposedDetectionGeometries[0][4][y]

      for (x in 0 until transposeddetectionScores[0][0][0].size) {
        if (detectionScoreData[x] < 0.5) {
          continue
        }

        // Compute the rotated bounding boxes and confiences (heavily based on OpenCV example):
        // https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py
        val offsetX = x * 4.0
        val offsetY = y * 4.0

        val h = detectionGeometryX0Data[x] + detectionGeometryX2Data[x]
        val w = detectionGeometryX1Data[x] + detectionGeometryX3Data[x]

        val angle = detectionRotationAngleData[x]
        val cos = Math.cos(angle.toDouble())
        val sin = Math.sin(angle.toDouble())

        val offset =
          Point(
            offsetX + cos * detectionGeometryX1Data[x] + sin * detectionGeometryX2Data[x],
            offsetY - sin * detectionGeometryX1Data[x] + cos * detectionGeometryX2Data[x]
          )
        val p1 = Point(-sin * h + offset.x, -cos * h + offset.y)
        val p3 = Point(-cos * w + offset.x, sin * w + offset.y)
        val center = Point(0.5 * (p1.x + p3.x), 0.5 * (p1.y + p3.y))

        val textDetection =
          RotatedRect(
            center,
            Size(w.toDouble(), h.toDouble()),
            (-1 * angle * 180.0 / Math.PI)
          )
        detectedRotatedRects.add(textDetection)
        detectedConfidences.add(detectionScoreData[x])
      }
    }

    detectedConfidencesMat = if (detectedConfidences.size > 0) MatOfFloat(vector_float_to_Mat(detectedConfidences)) else MatOfFloat()
    boundingBoxesMat = if (detectedRotatedRects.size > 0) MatOfRotatedRect(vector_RotatedRect_to_Mat(detectedRotatedRects)) else MatOfRotatedRect()

    NMSBoxesRotated(
      boundingBoxesMat,
      detectedConfidencesMat,
      detectionConfidenceThreshold.toFloat(),
      detectionNMSThreshold.toFloat(),
      indicesMat
    )
  }

  private fun recognizeTexts(
    data: Bitmap,
    displayBox: BoundingBox?,
    boundingBoxesMat: MatOfRotatedRect,
    indicesMat: MatOfInt
  ): Bitmap {
    val bitmapWithBoundingBoxes = data.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(bitmapWithBoundingBoxes)
    val paint = Paint()
    paint.style = Paint.Style.STROKE
    paint.strokeWidth = 5.toFloat()
    paint.setColor(Color.GREEN)

    val paint2 = Paint()
    paint2.style = Paint.Style.STROKE
    paint2.strokeWidth = 5.toFloat()
    paint2.setColor(Color.BLUE)

    if (displayBox != null)
      canvas.drawRect(
        displayBox.x_min*ratioWidth,
        displayBox.y_min*ratioHeight,
        displayBox.x_max*ratioWidth,
        displayBox.y_max*ratioHeight,
        paint2)

    // Find the text box has the maximum IOU with display box
    val ious:MutableList<Float> = ArrayList()
    for (i in indicesMat.toArray()) {
      val boundingBox = boundingBoxesMat.toArray()[i]
      val srcVertices = ArrayList<Point>()

      val boundingBoxPointsMat = Mat()
      boxPoints(boundingBox, boundingBoxPointsMat)

      for (j in 0 until 4) {
        srcVertices.add(
          Point(
            boundingBoxPointsMat.get(j, 0)[0],
            boundingBoxPointsMat.get(j, 1)[0]
          )
        )
      }

      val srcVerticesMat =
        MatOfPoint2f(srcVertices[0], srcVertices[1], srcVertices[2], srcVertices[3])

      val r = srcVerticesMat.toArray()
      val x_min = minOf(r[0].x, r[1].x, r[2].x, r[3].x).toFloat()
      val y_min = minOf(r[0].y, r[1].y, r[2].y, r[3].y).toFloat()
      val x_max = maxOf(r[0].x, r[1].x, r[2].x, r[3].x).toFloat()
      val y_max = maxOf(r[0].y, r[1].y, r[2].y, r[3].y).toFloat()

      val textBox = BoundingBox()
      textBox.x_min = x_min
      textBox.y_min = y_min
      textBox.x_max = x_max
      textBox.y_max = y_max

      val iou = if (displayBox != null) iou(textBox, displayBox) else 0.0f
      ious += iou
    }

    var best_iou_idx =  ious.withIndex().maxByOrNull { it.value }?.index
    if (best_iou_idx == null)
      best_iou_idx = 0

    val boundingBox = boundingBoxesMat.toArray()[indicesMat.toArray()[best_iou_idx]]
    boundingBox.size.height += 10.toDouble()
    boundingBox.size.width += 10.toDouble()
    val targetVertices = ArrayList<Point>()
    targetVertices.add(Point(0.toDouble(), (recognitionImageHeight - 1).toDouble()))
    targetVertices.add(Point(0.toDouble(), 0.toDouble()))
    targetVertices.add(Point((recognitionImageWidth - 1).toDouble(), 0.toDouble()))
    targetVertices.add(
      Point((recognitionImageWidth - 1).toDouble(), (recognitionImageHeight - 1).toDouble())
    )

    val srcVertices = ArrayList<Point>()

    val boundingBoxPointsMat = Mat()
    boxPoints(boundingBox, boundingBoxPointsMat)
    for (j in 0 until 4) {
      srcVertices.add(
        Point(
          boundingBoxPointsMat.get(j, 0)[0] * ratioWidth,
          boundingBoxPointsMat.get(j, 1)[0] * ratioHeight
        )
      )
      if (j != 0) {
        canvas.drawLine(
          (boundingBoxPointsMat.get(j, 0)[0] * ratioWidth).toFloat(),
          (boundingBoxPointsMat.get(j, 1)[0] * ratioHeight).toFloat(),
          (boundingBoxPointsMat.get(j - 1, 0)[0] * ratioWidth).toFloat(),
          (boundingBoxPointsMat.get(j - 1, 1)[0] * ratioHeight).toFloat(),
          paint
        )
      }
    }
    canvas.drawLine(
      (boundingBoxPointsMat.get(0, 0)[0] * ratioWidth).toFloat(),
      (boundingBoxPointsMat.get(0, 1)[0] * ratioHeight).toFloat(),
      (boundingBoxPointsMat.get(3, 0)[0] * ratioWidth).toFloat(),
      (boundingBoxPointsMat.get(3, 1)[0] * ratioHeight).toFloat(),
      paint
    )

    val srcVerticesMat =
      MatOfPoint2f(srcVertices[0], srcVertices[1], srcVertices[2], srcVertices[3])
    val targetVerticesMat =
      MatOfPoint2f(targetVertices[0], targetVertices[1], targetVertices[2], targetVertices[3])
    val rotationMatrix = getPerspectiveTransform(srcVerticesMat, targetVerticesMat)
    val recognitionBitmapMat = Mat()
    val srcBitmapMat = Mat()
    bitmapToMat(data, srcBitmapMat)
    warpPerspective(
      srcBitmapMat,
      recognitionBitmapMat,
      rotationMatrix,
      Size(recognitionImageWidth.toDouble(), recognitionImageHeight.toDouble())
    )

    val recognitionBitmap =
      ImageUtils.createEmptyBitmap(
        recognitionImageWidth,
        recognitionImageHeight,
        0,
        Bitmap.Config.ARGB_8888
      )
    matToBitmap(recognitionBitmapMat, recognitionBitmap)

    val recognitionTensorImage =
      ImageUtils.bitmapToTensorImageForRecognition(
        recognitionBitmap,
        recognitionImageWidth,
        recognitionImageHeight,
        recognitionImageMean,
        recognitionImageStd
      )

    recognitionResult.rewind()
    recognitionInterpreter.run(recognitionTensorImage.buffer, recognitionResult)

    var recognizedText = ""
    for (k in 0 until recognitionModelOutputSize) {
      var alphabetIndex = recognitionResult.getInt(k * 8)
      if (alphabetIndex in 0..alphabets.length - 1)
        recognizedText = recognizedText + alphabets[alphabetIndex]
    }

    Log.d("Recognition result:", recognizedText)
    if (recognizedText != "")
      ocrResults = recognizedText

    return bitmapWithBoundingBoxes
  }
  // base:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java
  @Throws(IOException::class)
  private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(modelFile)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    fileDescriptor.close()
    return retFile
  }

  @Throws(IOException::class)
  private fun getInterpreter(
    context: Context,
    modelName: String
  ): Interpreter {
    val tfliteOptions = Interpreter.Options()
    tfliteOptions.setNumThreads(numberThreads)

    return Interpreter(loadModelFile(context, modelName), tfliteOptions)
  }

  override fun close() {
    detectionInterpreter.close()
    recognitionInterpreter.close()
  }

  /**
   * Given two bounding , compute the intersecton over union of the two, which is area of overlap divided by area of union
   * Returns: iou: `float32`Intersection over union value with a range `0~1`
   */
  private fun iou(boxA: BoundingBox, boxB: BoundingBox): Float {
    // determine the (x, y)-coordinates of the intersection rectangle
    val xA = maxOf(boxA.x_min, boxB.x_min)
    val yA = maxOf(boxA.y_min, boxB.y_min)
    val xB = minOf(boxA.x_max, boxB.x_max)
    val yB = minOf(boxA.y_max, boxB.y_max)

    // compute the area of intersection rectangle
    val interArea = abs(maxOf(xB - xA, 0.0f) * maxOf(yB - yA, 0.0f))
    if (interArea.toInt() == 0)
      return 0.0f

    // compute the area of both the prediction and ground-truth
    val boxAArea = abs((boxA.x_max - boxA.x_min) * (boxA.y_max - boxA.y_min))
    val boxBArea = abs((boxB.x_max - boxB.x_min) * (boxB.y_max - boxB.y_min))

    val iou = interArea / (boxAArea + boxBArea - interArea)
    return iou
  }

  /**
   * Rectangle for representing bounding box detected by display detection
   */
  class BoundingBox {
    var x_min: Float = 0.0f
    var y_min: Float = 0.0f
    var x_max: Float = 0.0f
    var y_max: Float = 0.0f
  }

  companion object {
    const val TAG = "TfLiteMeeterReaderSDK"
    private const val displayDetectionModel = "display_detection.tflite"
    private const val textDetectionModel = "reading_detectionV2.tflite"
    private const val textRecognitionModel = "reading_ocrV3.tflite"
    private const val numberThreads = 4
    private const val alphabets = "0123456789."
    private const val displayDetectionImageMean = 0.toFloat()
    private const val displayDetectionImageStd = 255.toFloat()
    private const val displayImageSize = 257
    private const val detectionImageHeight = 320
    private const val detectionImageWidth = 320
    private val detectionOutputNumRows = 80
    private val detectionOutputNumCols = 80
    private val detectionConfidenceThreshold = 0.1
    private val detectionNMSThreshold = 0.2
    private const val recognitionImageHeight = 31
    private const val recognitionImageWidth = 200
    private const val recognitionImageMean = 0.toFloat()
    private const val recognitionImageStd = 255.toFloat()
    private const val recognitionModelOutputSize = 48
  }
}
