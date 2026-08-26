package com.pes.facialparalysis.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class ResNetFeatureExtractor(context: Context) {

    private val interpreter: Interpreter

    companion object {
        private const val INPUT_SIZE = 224
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    init {
        val assetFileDescriptor = context.assets.openFd("resnet50_feat_float16.tflite")
        val fileInputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

        interpreter = Interpreter(modelBuffer)

        // TEMP DEBUG: confirm actual input tensor shape. Remove once confirmed.
        android.util.Log.d("ResNet", "Input shape: ${interpreter.getInputTensor(0).shape().contentToString()}")
    }

    /** Resizes bitmap to 224x224, normalizes with ImageNet mean/std, returns 2048-dim embedding */
    fun extract(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = bitmapToInputBuffer(resized)

        // Actual output shape after onnx2tf conversion: [1, 1, 1, 2048]
        val output = Array(1) { Array(1) { Array(1) { FloatArray(2048) } } }
        interpreter.run(inputBuffer, output)

        return FloatArray(2048) { i -> output[0][0][0][i] }
    }

    private fun bitmapToInputBuffer(bitmap: Bitmap): ByteBuffer {
        // NHWC format: [1, 224, 224, 3] — required by the onnx2tf-converted TFLite model
        val buffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            buffer.putFloat((r - MEAN[0]) / STD[0])
            buffer.putFloat((g - MEAN[1]) / STD[1])
            buffer.putFloat((b - MEAN[2]) / STD[2])
        }

        buffer.rewind()
        return buffer
    }

    fun close() {
        interpreter.close()
    }
}