package com.pes.facialparalysis.reporting

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.pdf.PdfDocument
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class ReportData(
    val patientName: String,
    val timestamp: Long,
    val grade: Int,
    val gradeLabel: String,
    val confidence: Double,
    val explanation: String
)

/**
 * Builds a single-page A4 PDF. `imageFile` is only drawn if non-null — pass null
 * when the user declined via ConsentDialog and the report goes out without the photo.
 */
object PdfReportGenerator {

    private const val PAGE_W = 595 // A4 at 72dpi
    private const val PAGE_H = 842
    private const val MARGIN = 40f

    fun generate(context: Context, data: ReportData, imageFile: File?): File {
        val document = PdfDocument()
        val pageInfo = PdfDocument.PageInfo.Builder(PAGE_W, PAGE_H, 1).create()
        val page = document.startPage(pageInfo)
        val canvas: Canvas = page.canvas

        val titlePaint = Paint().apply { color = Color.BLACK; textSize = 20f; isFakeBoldText = true }
        val labelPaint = Paint().apply { color = Color.DKGRAY; textSize = 11f }
        val valuePaint = Paint().apply { color = Color.BLACK; textSize = 13f }
        val bodyPaint = Paint().apply { color = Color.BLACK; textSize = 12f }

        var y = MARGIN + 10f
        canvas.drawText("Facial Assessment Report", MARGIN, y, titlePaint)
        y += 30f

        val dateStr = SimpleDateFormat("dd MMM yyyy, HH:mm", Locale.getDefault()).format(Date(data.timestamp))
        y = drawField(canvas, "Patient", data.patientName, MARGIN, y, labelPaint, valuePaint)
        y = drawField(canvas, "Date", dateStr, MARGIN, y, labelPaint, valuePaint)
        y = drawField(canvas, "Grade", "G${data.grade} - ${data.gradeLabel}", MARGIN, y, labelPaint, valuePaint)
        y = drawField(
            canvas, "Confidence", "${"%.0f".format(data.confidence * 100)}%",
            MARGIN, y, labelPaint, valuePaint
        )
        y += 14f

        canvas.drawText("Analysis", MARGIN, y, labelPaint)
        y += 16f
        y = drawWrapped(canvas, data.explanation, MARGIN, y, PAGE_W - 2 * MARGIN, bodyPaint)
        y += 20f

        if (imageFile != null && imageFile.exists()) {
            val bmp = BitmapFactory.decodeFile(imageFile.absolutePath)
            if (bmp != null) {
                val maxW = PAGE_W - 2 * MARGIN
                val maxH = PAGE_H - y - MARGIN
                val scale = minOf(maxW / bmp.width, maxH / bmp.height, 1f)
                val w = bmp.width * scale
                val h = bmp.height * scale
                val dest = RectF(MARGIN, y, MARGIN + w, y + h)
                canvas.drawBitmap(bmp, null, dest, null)
                y += h + 10f
            }
        } else {
            val notePaint = Paint().apply { color = Color.GRAY; textSize = 10f; textSkewX = -0.25f }
            canvas.drawText("(Report generated without photo, per patient/user consent.)", MARGIN, y, notePaint)
        }

        document.finishPage(page)

        val outDir = File(context.cacheDir, "reports").apply { mkdirs() }
        val outFile = File(outDir, "report_${data.patientName.filter { it.isLetterOrDigit() }}_${data.timestamp}.pdf")
        FileOutputStream(outFile).use { document.writeTo(it) }
        document.close()
        return outFile
    }

    /** Content URI for sharing/opening the PDF via FileProvider (declare a provider in the manifest). */
    fun uriFor(context: Context, file: File) =
        FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)

    private fun drawField(
        canvas: Canvas, label: String, value: String, x: Float, y: Float,
        labelPaint: Paint, valuePaint: Paint
    ): Float {
        canvas.drawText(label, x, y, labelPaint)
        canvas.drawText(value, x + 90f, y, valuePaint)
        return y + 20f
    }

    private fun drawWrapped(
        canvas: Canvas, text: String, x: Float, startY: Float, maxWidth: Float, paint: Paint
    ): Float {
        var y = startY
        val words = text.split(" ")
        var line = StringBuilder()
        for (word in words) {
            val candidate = if (line.isEmpty()) word else "$line $word"
            if (paint.measureText(candidate) > maxWidth) {
                canvas.drawText(line.toString(), x, y, paint)
                y += 16f
                line = StringBuilder(word)
            } else {
                line = StringBuilder(candidate)
            }
        }
        if (line.isNotEmpty()) {
            canvas.drawText(line.toString(), x, y, paint)
            y += 16f
        }
        return y
    }
}