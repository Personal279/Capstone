package com.pes.facialparalysis.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.pes.facialparalysis.HomeScreen
import com.pes.facialparalysis.ui.screens.CaptureScreen
import com.pes.facialparalysis.ui.screens.HistoryScreen
import com.pes.facialparalysis.ui.screens.PatientSelectionScreen
import com.pes.facialparalysis.ui.screens.UploadPreviewScreen
import com.pes.facialparalysis.ui.screens.VideoCaptureScreen
import com.pes.facialparalysis.ui.screens.ResultScreen
@Composable
fun AppNavHost(
    navController: NavHostController = rememberNavController()
) {
    NavHost(
        navController = navController,
        startDestination = Screen.PatientSelection.route
    ) {

        composable(Screen.PatientSelection.route) {
            PatientSelectionScreen(
                onPatientSelected = {
                    navController.navigate(Screen.Home.route)
                }
            )
        }

        composable(Screen.Home.route) {
            HomeScreen(
                onUploadClick = {
                    navController.navigate(Screen.UploadPreview.route)
                },
                onRecordClick = {
                    navController.navigate(Screen.VideoCapture.route)
                },
                onCameraClick = {
                    navController.navigate(Screen.Capture.route)
                },
                onHistoryClick = {
                    navController.navigate(Screen.History.route)
                },
                onSwitchPatientClick = {
                    navController.navigate(Screen.PatientSelection.route)
                }
            )
        }

        composable(Screen.UploadPreview.route) {
            UploadPreviewScreen(
                onConfirmed = {
                    navController.navigate(Screen.Result.route)
                },
                onCancelled = {
                    navController.popBackStack()
                }
            )
        }

        composable(Screen.Capture.route) {
            CaptureScreen(
                onCaptured = {
                    navController.navigate(Screen.Result.route)
                }
            )
        }

        composable(Screen.VideoCapture.route) {
            VideoCaptureScreen(
                onRecorded = {
                    navController.navigate(Screen.Result.route)
                }
            )
        }

        composable(Screen.Result.route) {
            ResultScreen(
                onDone = {
                    navController.navigate(Screen.Home.route) {
                        popUpTo(Screen.Home.route) {
                            inclusive = true
                        }
                    }
                }
            )
        }

        composable(Screen.History.route) {
            HistoryScreen()
        }
    }
}