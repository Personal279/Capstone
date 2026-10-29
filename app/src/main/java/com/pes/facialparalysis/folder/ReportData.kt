package com.pes.facialparalysis.report

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.pes.facialparalysis.data.AssessmentRecord
import org.json.JSONObject
import java.io.File
import kotlin.math.max

/** Everything the PDF and the record detail screen need, built from the database only. */
data class ReportData(
    val patientName: String,
    val record: AssessmentRecord,
    val initialImagePath: String?,
    val latestImagePath: String?
)

val regionDisplayNames = mapOf(
    "eye" to "Eye region",
    "mouth" to "Mouth region",
    "brow" to "Eyebrow region",
    "cheek" to "Cheek region",
    "jaw" to "Lower face / jaw region"
)

/** Parses {"eye":0.3,...}. Returns empty map for blank/invalid JSON. */
fun parseDoubleMap(json: String): Map<String, Double> {
    if (json.isBlank()) return emptyMap()
    return try {
        val obj = JSONObject(json)
        obj.keys().asSequence().associateWith { obj.getDouble(it) }
    } catch (e: Exception) {
        emptyMap()
    }
}

/** Parses {"1":0.02,"2":0.1,...} into grade -> probability. */
fun parseProbabilities(json: String): Map<Int, Double> =
    parseDoubleMap(json).mapNotNull { (k, v) -> k.toIntOrNull()?.let { it to v } }.toMap()

/** Converts LIME importances to percentage shares that sum to 100. */
fun relativeShares(map: Map<String, Double>): Map<String, Double> {
    val total = map.values.sumOf { kotlin.math.abs(it) }
    if (total <= 0.0) return emptyMap()
    return map.mapValues { kotlin.math.abs(it.value) / total * 100.0 }
}

/** Decodes a downsampled bitmap (longest side about [maxDim] px). Returns null if missing/unreadable. */
fun decodeSampledBitmap(path: String?, maxDim: Int): Bitmap? {
    if (path == null) return null
    val file = File(path)
    if (!file.exists()) return null
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(path, bounds)
    var sample = 1
    while (max(bounds.outWidth, bounds.outHeight) / sample > maxDim) sample *= 2
    return BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = sample })
}