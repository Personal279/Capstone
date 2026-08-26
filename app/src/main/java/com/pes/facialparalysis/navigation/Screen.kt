package com.pes.facialparalysis.navigation

sealed class Screen(val route: String) {
    object PatientSelection : Screen("patient_selection")
    object Home : Screen("home")
    object Capture : Screen("capture")
    object VideoCapture : Screen("video_capture")
    object Result : Screen("result")
    object History : Screen("history")
    object UploadPreview : Screen("upload_preview")
}