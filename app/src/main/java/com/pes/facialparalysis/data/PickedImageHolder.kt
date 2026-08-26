package com.pes.facialparalysis.data

import android.net.Uri

object PickedImageHolder {
    var uri: Uri? = null

    fun clear() {
        uri = null
    }
}