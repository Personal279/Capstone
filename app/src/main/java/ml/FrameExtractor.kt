package com.pes.facialparalysis.ml

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import java.io.File

object FrameExtractor {

    /** Extracts up to maxFrames evenly spaced frames from a video file */
    fun extract(videoFile: File, maxFrames: Int = 10): List<Bitmap> {
        val retriever = MediaMetadataRetriever()
        val frames = mutableListOf<Bitmap>()

        try {
            retriever.setDataSource(videoFile.absolutePath)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L

            if (durationMs <= 0) return emptyList()

            val stepMs = durationMs / (maxFrames + 1)
            for (i in 1..maxFrames) {
                val timeUs = (stepMs * i) * 1000
                val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                if (frame != null) frames.add(frame)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            retriever.release()
        }

        return frames
    }
}