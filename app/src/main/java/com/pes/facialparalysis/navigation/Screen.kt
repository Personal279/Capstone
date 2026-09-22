package com.pes.facialparalysis.navigation

sealed class Screen(val route: String) {
    object Splash : Screen("splash")
    object Onboarding : Screen("onboarding")
    object PatientSelection : Screen("patient_selection")
    object Home : Screen("home")
    object Capture : Screen("capture")
    object VideoCapture : Screen("video_capture")
    object Result : Screen("result")
    object History : Screen("history")
    object UploadPreview : Screen("upload_preview")
    object XaiExplanation : Screen("xai_explanation")
    object DigitalTwin : Screen("digital_twin")
}