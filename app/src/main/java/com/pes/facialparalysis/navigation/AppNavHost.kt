package com.pes.facialparalysis.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.pes.facialparalysis.HomeScreen
import com.pes.facialparalysis.ui.screens.CaptureScreen
import com.pes.facialparalysis.ui.screens.HistoryScreen
import com.pes.facialparalysis.ui.screens.PatientSelectionScreen
import com.pes.facialparalysis.ui.screens.UploadPreviewScreen
import com.pes.facialparalysis.ui.screens.VideoCaptureScreen
import com.pes.facialparalysis.ui.screens.ResultScreen
import com.pes.facialparalysis.ui.screens.XaiExplanationScreen
import com.pes.facialparalysis.ui.screens.DigitalTwinScreen
import com.pes.facialparalysis.ui.screens.SplashScreen
import com.pes.facialparalysis.ui.screens.OnboardingScreen
import com.pes.facialparalysis.ui.screens.RecordDetailScreen

@Composable
fun AppNavHost(
    navController: NavHostController = rememberNavController()
) {
    NavHost(
        navController = navController,
        startDestination = Screen.Splash.route
    ) {

        composable(Screen.Splash.route) {
            SplashScreen(
                onBeginAssessment = {
                    navController.navigate(Screen.PatientSelection.route) {
                        popUpTo(Screen.Splash.route) { inclusive = true }
                    }
                }
            )
        }

        composable(Screen.Onboarding.route) {
            OnboardingScreen(
                onStartGuidedCapture = {
                    navController.navigate(Screen.Capture.route)
                }
            )
        }

        composable(Screen.PatientSelection.route) {
            PatientSelectionScreen(
                onPatientSelected = {
                    navController.navigate(Screen.Home.route)
                },
                onViewDigitalTwin = {
                    navController.navigate(Screen.DigitalTwin.route)
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
                    navController.navigate(Screen.Onboarding.route)
                },
                onHistoryClick = {
                    navController.navigate(Screen.History.route)
                },
                onDigitalTwinClick = {
                    navController.navigate(Screen.DigitalTwin.route)
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
                },
                onViewExplanation = {
                    navController.navigate(Screen.XaiExplanation.route)
                },
                onViewDigitalTwin = {
                    navController.navigate(Screen.DigitalTwin.route)
                },
                onOpenReport = { recordId ->
                    navController.navigate(Screen.RecordDetail.create(recordId))
                }
            )
        }

        composable(Screen.XaiExplanation.route) {
            XaiExplanationScreen(
                onBack = { navController.popBackStack() }
            )
        }

        composable(Screen.DigitalTwin.route) {
            DigitalTwinScreen(
                onBack = { navController.popBackStack() }
            )
        }

        composable(Screen.History.route) {
            HistoryScreen(
                onRecordClick = { record ->
                    navController.navigate(Screen.RecordDetail.create(record.id))
                },
                onStartNewAssessment = {
                    navController.popBackStack()
                }
            )
        }

        composable(
            route = Screen.RecordDetail.route,
            arguments = listOf(navArgument("recordId") { type = NavType.IntType })
        ) { entry ->
            RecordDetailScreen(
                recordId = entry.arguments?.getInt("recordId") ?: -1,
                onBack = { navController.popBackStack() }
            )
        }
    }
}