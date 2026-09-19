package com.pes.facialparalysis.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil

class MlpClassifier(context: Context) {

    private val interpreter: Interpreter = Interpreter(
        FileUtil.loadMappedFile(context, "severity_model_float32.tflite")
    )

    private val imageMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val imageStd = floatArrayOf(0.229f, 0.224f, 0.225f)

    data class PredictionResult(val predictedGrade: Int, val probabilities: Map<Int, Double>)

    fun predict(bitmap: Bitmap): PredictionResult {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resized.getPixel(x, y)
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f
                input[0][y][x][0] = (r - imageMean[0]) / imageStd[0]
                input[0][y][x][1] = (g - imageMean[1]) / imageStd[1]
                input[0][y][x][2] = (b - imageMean[2]) / imageStd[2]
            }
        }

        val output = Array(1) { FloatArray(5) }
        interpreter.run(input, output)

        val classes = intArrayOf(1, 2, 3, 4, 5)
        val probs = output[0]
        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val predictedGrade = classes[maxIdx]
        val probMap = classes.indices.associate { classes[it] to probs[it].toDouble() }

        return PredictionResult(predictedGrade, probMap)
    }

    fun close() = interpreter.close()
}