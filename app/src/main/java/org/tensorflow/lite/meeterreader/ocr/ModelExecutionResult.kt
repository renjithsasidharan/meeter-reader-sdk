/*
 * Created by Renjith Sasidharan on 21/02/22, 11:57 PM
 * renjithks@gmail.com
 * Last modified 21/02/22, 11:57 PM
 * Copyright (c) 2022.
 * All rights reserved.
 */


package org.tensorflow.lite.meeterreader.ocr

import android.graphics.Bitmap

/**
 * The result of OCR model containing meeter reading
 * @property bitmapResult the annotated image
 * @property executionLog contains error messages or info messages
 * @property reading the meeter reading
 */
data class ModelExecutionResult(
  val bitmapResult: Bitmap, // Annotated image
  val executionLog: String, // Info, logs and error messages
  val reading: String? // Meeter reading
)