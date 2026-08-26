package com.pes.facialparalysis.data

import android.graphics.Bitmap

object CapturedImageHolder {
    var bitmap: Bitmap? = null

    fun clear() {
        bitmap?.recycle()
        bitmap = null
    }
}