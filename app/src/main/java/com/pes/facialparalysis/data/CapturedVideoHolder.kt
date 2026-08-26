package com.pes.facialparalysis.data

import java.io.File

object CapturedVideoHolder {
    var videoFile: File? = null

    fun clear() {
        videoFile?.delete()
        videoFile = null
    }
}