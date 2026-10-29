package com.pes.facialparalysis.report

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.text.StaticLayout
import android.text.TextPaint
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.min

private const val PAGE_W = 595 // A4 @ 72dpi
private const val PAGE_H = 842
private const val MARGIN = 40f

private const val INK = 0xFF1B2B2E.toInt()
private const val MUTED = 0xFF5C6B6E.toInt()
private const val TEAL = 0xFF0F6E7A.toInt()
private const val LINE = 0xFFDCE5E6.toInt()

private fun textPaint(size: Float, bold: Boolean = false, colorInt: Int = INK): TextPaint {
    val p = TextPaint(Paint.ANTI_ALIAS_FLAG)
    p.textSize = size
    p.color = colorInt
    p.typeface = if (bold) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
    return p
}

private class PdfWriter(private val doc: PdfDocument) {
    private var pageNo = 0
    private var page: PdfDocument.Page? = null
    private lateinit var canvas: Canvas
    var y = MARGIN
        private set

    private val footerPaint = textPaint(8f, colorInt = MUTED)
    private val linePaint = Paint().apply { color = LINE; strokeWidth = 1f }
    private val borderPaint = Paint().apply { color = LINE; style = Paint.Style.STROKE; strokeWidth = 1f }
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    val body = textPaint(10.5f)
    val small = textPaint(9f, colorInt = MUTED)
    val bold = textPaint(10.5f, bold = true)
    val title = textPaint(20f, bold = true, colorInt = TEAL)
    val heading = textPaint(13f, bold = true, colorInt = TEAL)

    fun start() = newPage()

    private fun newPage() {
        closePage()
        pageNo++
        page = doc.startPage(PdfDocument.PageInfo.Builder(PAGE_W, PAGE_H, pageNo).create())
        canvas = page!!.canvas
        y = MARGIN
    }

    private fun closePage() {
        page?.let {
            canvas.drawText(
                "AI-generated estimate - not a medical diagnosis. Page $pageNo",
                MARGIN, PAGE_H - 22f, footerPaint
            )
            doc.finishPage(it)
        }
        page = null
    }

    fun finish() = closePage()

    private fun ensure(h: Float) {
        if (y + h > PAGE_H - MARGIN - 16f) newPage()
    }

    fun gap(h: Float) { y += h }

    fun paragraph(text: String, paint: TextPaint, after: Float = 6f) {
        if (text.isBlank()) return
        val width = (PAGE_W - 2 * MARGIN).toInt()
        val layout = StaticLayout.Builder.obtain(text, 0, text.length, paint, width)
            .setLineSpacing(2f, 1f)
            .build()
        for (i in 0 until layout.lineCount) {
            val lineH = (layout.getLineBottom(i) - layout.getLineTop(i)).toFloat()
            ensure(lineH)
            val baseline = y + (layout.getLineBaseline(i) - layout.getLineTop(i))
            canvas.drawText(text, layout.getLineStart(i), layout.getLineEnd(i), MARGIN, baseline, paint)
            y += lineH
        }
        y += after
    }

    fun section(text: String) {
        ensure(44f)
        gap(10f)
        paragraph(text, heading, after = 2f)
        canvas.drawLine(MARGIN, y, PAGE_W - MARGIN, y, linePaint)
        y += 8f
    }

    fun row(label: String, value: String) {
        ensure(18f)
        canvas.drawText(label, MARGIN, y + 11f, body)
        val w = bold.measureText(value)
        canvas.drawText(value, PAGE_W - MARGIN - w, y + 11f, bold)
        y += 18f
    }

    fun imagePair(left: Bitmap?, leftLabel: String, right: Bitmap?, rightLabel: String) {
        val gapW = 16f
        val boxW = (PAGE_W - 2 * MARGIN - gapW) / 2f
        val boxH = boxW * 4f / 3f
        ensure(boxH + 26f)
        listOf(left to leftLabel, right to rightLabel).forEachIndexed { i, (bmp, label) ->
            val x = MARGIN + i * (boxW + gapW)
            canvas.drawText(label, x, y + 9f, small)
            val top = y + 14f
            canvas.drawRect(RectF(x, top, x + boxW, top + boxH), borderPaint)
            if (bmp != null) {
                val scale = min(boxW / bmp.width, boxH / bmp.height)
                val w = bmp.width * scale
                val h = bmp.height * scale
                val dx = x + (boxW - w) / 2f
                val dy = top + (boxH - h) / 2f
                canvas.drawBitmap(bmp, null, RectF(dx, dy, dx + w, dy + h), bitmapPaint)
            }
        }
        y += boxH + 26f
    }
}

object ReportExporter {

    private fun fileName(data: ReportData): String {
        val safe = data.patientName.replace(Regex("[^A-Za-z0-9]+"), "_").trim('_').ifEmpty { "patient" }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmm", Locale.US).format(Date(data.record.timestamp))
        return "NeuroVision_Report_${safe}_$stamp.pdf"
    }

    /**
     * Builds the PDF in the app cache. Face images are drawn ONLY when [includeImage] is true,
     * i.e. when the user ticked the consent checkbox for this specific export.
     */
    suspend fun createPdf(context: Context, data: ReportData, includeImage: Boolean): File =
        withContext(Dispatchers.IO) {
            val dir = File(context.cacheDir, "reports").apply {
                mkdirs()
                listFiles()?.forEach { it.delete() } // never leave old reports lying around
            }
            val file = File(dir, fileName(data))
            val doc = PdfDocument()
            var initial: Bitmap? = null
            var latest: Bitmap? = null
            try {
                if (includeImage) {
                    initial = decodeSampledBitmap(data.initialImagePath, 900)
                    latest = decodeSampledBitmap(data.latestImagePath, 900)
                }
                val r = data.record
                val w = PdfWriter(doc)
                w.start()

                w.paragraph("NeuroVision AI", w.title, after = 2f)
                w.paragraph("Facial paralysis assessment report", w.bold, after = 8f)
                w.row("Patient", data.patientName)
                w.row("Assessment date", SimpleDateFormat("MMM d, yyyy - h:mm a", Locale.getDefault()).format(Date(r.timestamp)))
                w.row("Report generated", SimpleDateFormat("MMM d, yyyy - h:mm a", Locale.getDefault()).format(Date()))

                w.section("House-Brackmann grade")
                w.row("Grade", "Grade ${r.grade} - ${r.gradeLabel}")
                w.row("Assessment confidence", "%.0f%%".format(r.confidence * 100))
                w.row("Overall asymmetry index", "%.1f%%".format(r.faiValue))

                val probs = parseProbabilities(r.probabilitiesJson)
                if (probs.isNotEmpty()) {
                    w.section("Confidence by grade")
                    probs.toSortedMap().forEach { (g, p) -> w.row("Grade $g", "%.1f%%".format(p * 100)) }
                }

                w.section("Measured asymmetry by region")
                listOf(
                    "eye" to r.eyeAsymmetry, "mouth" to r.mouthAsymmetry, "brow" to r.browAsymmetry,
                    "cheek" to r.cheekAsymmetry, "jaw" to r.jawAsymmetry
                ).sortedByDescending { it.second }.forEach { (key, v) ->
                    w.row(regionDisplayNames[key] ?: key, "%.1f%%".format(v))
                }
                w.paragraph(
                    "Asymmetry % = difference between the left- and right-side landmark measurement for that region.",
                    w.small
                )

                w.section("AI explanation - why this grade?")
                if (r.limeSummary.isNotBlank()) w.paragraph(r.limeSummary, w.bold)
                val shares = relativeShares(parseDoubleMap(r.limeImportanceJson))
                if (shares.isNotEmpty()) {
                    w.paragraph("Relative influence of each region on the AI's prediction (LIME):", w.small, after = 2f)
                    shares.entries.sortedByDescending { it.value }.forEach { (k, v) ->
                        w.row(regionDisplayNames[k] ?: k, "%.0f%%".format(v))
                    }
                    w.gap(4f)
                }
                if (r.topRegionsCsv.isNotBlank()) {
                    w.paragraph("Landmark measurement notes:", w.small, after = 2f)
                    r.topRegionsCsv.split(",").filter { it.isNotBlank() }.forEach { w.paragraph("- ${it.trim()}", w.body, after = 1f) }
                    w.gap(4f)
                }
                w.paragraph(r.xaiExplanation, w.body)

                if (includeImage && (initial != null || latest != null)) {
                    w.section("Images (included with your permission)")
                    w.imagePair(initial, "First test (initial)", latest, "Most recent test")
                } else {
                    w.paragraph("Images were not included in this report.", w.small)
                }

                w.section("Limitations")
                w.paragraph(
                    "- This is a single automated estimate, not a clinical exam.\n" +
                            "- The model was trained on a limited dataset and may not generalize to every face, lighting condition or camera.\n" +
                            "- The explanation describes the model's behavior on this image; it is not a measurement of nerve or muscle function.\n" +
                            "- Only a qualified healthcare professional can diagnose facial paralysis.",
                    w.body
                )
                w.finish()
                file.outputStream().use { doc.writeTo(it) }
            } finally {
                doc.close()
                initial?.recycle()
                latest?.recycle()
            }
            file
        }

    /** Opens the Android share sheet for [file]. The user picks the destination every time. */
    fun share(context: Context, file: File) {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "application/pdf"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(send, "Share report").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    }

    /** Copies [file] into the public Downloads folder (Android 10+). Returns false if not possible. */
    suspend fun saveToDownloads(context: Context, file: File): Boolean = withContext(Dispatchers.IO) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return@withContext false
        try {
            val values = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, file.name)
                put(MediaStore.Downloads.MIME_TYPE, "application/pdf")
                put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
            }
            val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                ?: return@withContext false
            val out = context.contentResolver.openOutputStream(uri) ?: return@withContext false
            out.use { o -> file.inputStream().use { it.copyTo(o) } }
            true
        } catch (e: Exception) {
            false
        }
    }
}