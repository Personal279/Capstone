package com.pes.facialparalysis.data

import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream

/**
 * Keeps exactly two images per patient on disk:
 *  - baseline_<patientId>.jpg : the FIRST assessment photo ever taken for that patient
 *  - latest_<patientId>.jpg   : overwritten every time they take a new test
 *
 * File-based (not DB) so it doesn't require an AssessmentRecord/schema migration.
 * Point AppDatabase at these paths later if you want them in a report/history query.
 */
object ImageStore {

    private fun dir(context: Context): File =
        File(context.filesDir, "assessment_images").apply { mkdirs() }

    private fun baselineFile(context: Context, patientId: Int): File =
        File(dir(context), "baseline_$patientId.jpg")

    private fun latestFile(context: Context, patientId: Int): File =
        File(dir(context), "latest_$patientId.jpg")

    /**
     * Call this once per completed assessment (e.g. in ResultScreen after grading
     * succeeds). Saves as baseline only if one doesn't exist yet; always updates latest.
     */
    fun saveAssessmentImage(context: Context, patientId: Int, bitmap: Bitmap) {
        val baseline = baselineFile(context, patientId)
        if (!baseline.exists()) {
            writeJpeg(bitmap, baseline)
        }
        writeJpeg(bitmap, latestFile(context, patientId))
    }

    fun getBaseline(context: Context, patientId: Int): File? =
        baselineFile(context, patientId).takeIf { it.exists() }

    fun getLatest(context: Context, patientId: Int): File? =
        latestFile(context, patientId).takeIf { it.exists() }

    fun hasBaseline(context: Context, patientId: Int): Boolean =
        baselineFile(context, patientId).exists()

    private fun writeJpeg(bitmap: Bitmap, file: File) {
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
        }
    }
}