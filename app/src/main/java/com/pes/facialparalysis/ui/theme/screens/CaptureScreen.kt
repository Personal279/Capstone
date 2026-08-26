package com.pes.facialparalysis.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import com.pes.facialparalysis.data.CapturedImageHolder
import com.pes.facialparalysis.ui.theme.AppColors
import kotlinx.coroutines.delay

@Composable
fun CaptureScreen(onCaptured: () -> Unit) {
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    if (!hasPermission) {
        PermissionRationale(onRequestAgain = { permissionLauncher.launch(Manifest.permission.CAMERA) })
        return
    }

    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var isCapturing by remember { mutableStateOf(false) }
    var captureError by remember { mutableStateOf<String?>(null) }

    // Brief flash to confirm a shot was actually taken — matters when there's
    // no shutter sound/haptic guarantee across devices.
    var flashVisible by remember { mutableStateOf(false) }
    val flashAlpha by animateFloatAsState(
        targetValue = if (flashVisible) 0.85f else 0f,
        animationSpec = tween(durationMillis = if (flashVisible) 40 else 220),
        label = "shutterFlash"
    )

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    val capture = ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build()
                    imageCapture = capture

                    val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            ctx as androidx.lifecycle.LifecycleOwner,
                            cameraSelector,
                            preview,
                            capture
                        )
                    } catch (e: Exception) {
                        captureError = "Couldn't start the camera. Check that no other app is using it."
                    }
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            }
        )

        // Face alignment guide — the single most important addition. Grading
        // depends on comparable framing shot-to-shot, so give the user a
        // fixed oval to line their face up against instead of guessing.
        FaceGuideOverlay(modifier = Modifier.fillMaxSize())

        // Shutter flash feedback
        if (flashAlpha > 0f) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.White.copy(alpha = flashAlpha))
            )
        }

        // Top instruction bar — respects status bar via safe drawing insets
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.statusBars)
                .background(AppColors.TextPrimary.copy(alpha = 0.55f))
                .padding(horizontal = 20.dp, vertical = 14.dp)
        ) {
            Text(
                text = "Align your face with the guide",
                color = AppColors.TextOnPrimary,
                fontSize = 14.sp,
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )
            Text(
                text = "Neutral expression · good lighting · look straight ahead",
                color = AppColors.TextOnPrimary.copy(alpha = 0.8f),
                fontSize = 12.sp,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth().padding(top = 2.dp)
            )
        }

        // Capture button with real capturing/disabled state
        Button(
            onClick = {
                val capture = imageCapture ?: return@Button
                if (isCapturing) return@Button
                isCapturing = true
                captureError = null
                capture.takePicture(
                    ContextCompat.getMainExecutor(context),
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            val bitmap = imageProxyToBitmap(image)
                            image.close()
                            CapturedImageHolder.bitmap = bitmap
                            isCapturing = false
                            flashVisible = true
                            onCaptured()
                        }

                        override fun onError(exception: ImageCaptureException) {
                            isCapturing = false
                            captureError = "Capture failed — try again."
                        }
                    }
                )
            },
            enabled = !isCapturing,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .windowInsetsPadding(WindowInsets.navigationBars)
                .padding(bottom = 36.dp)
                .size(72.dp)
                .clip(CircleShape),
            shape = CircleShape,
            colors = ButtonDefaults.buttonColors(
                containerColor = AppColors.Primary,
                disabledContainerColor = AppColors.Primary.copy(alpha = 0.5f)
            ),
            contentPadding = PaddingValues(0.dp)
        ) {
            if (isCapturing) {
                CircularProgressIndicator(
                    color = AppColors.TextOnPrimary,
                    strokeWidth = 2.5.dp,
                    modifier = Modifier.size(28.dp)
                )
            } else {
                Icon(
                    imageVector = Icons.Filled.CameraAlt,
                    contentDescription = "Capture",
                    tint = AppColors.TextOnPrimary,
                    modifier = Modifier.size(28.dp)
                )
            }
        }

        captureError?.let { message ->
            LaunchedEffect(message) {
                delay(3000)
                captureError = null
            }
            Surface(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 130.dp)
                    .padding(horizontal = 24.dp),
                shape = RoundedCornerShape(10.dp),
                color = AppColors.TextPrimary
            ) {
                Text(
                    text = message,
                    color = AppColors.TextOnPrimary,
                    fontSize = 13.sp,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp)
                )
            }
        }
    }

    // Reset the flash after it fires
    LaunchedEffect(flashVisible) {
        if (flashVisible) {
            delay(60)
            flashVisible = false
        }
    }
}

/**
 * Dashed oval guide centered in the preview. Gives the user a consistent
 * target to line their face up against, which matters for downstream
 * landmark detection and symmetry grading.
 */
@Composable
private fun FaceGuideOverlay(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val guideWidth = size.width * 0.62f
        val guideHeight = size.height * 0.42f
        val center = Offset(size.width / 2f, size.height * 0.46f)

        drawOval(
            color = Color.White.copy(alpha = 0.85f),
            topLeft = Offset(center.x - guideWidth / 2f, center.y - guideHeight / 2f),
            size = Size(guideWidth, guideHeight),
            style = Stroke(
                width = 3.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(14f, 10f), 0f)
            )
        )
    }
}

@Composable
private fun PermissionRationale(onRequestAgain: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppColors.Background)
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(CircleShape)
                .background(AppColors.Primary.copy(alpha = 0.12f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Filled.PhotoCamera,
                contentDescription = null,
                tint = AppColors.Primary,
                modifier = Modifier.size(32.dp)
            )
        }
        Spacer(modifier = Modifier.height(20.dp))
        Text(
            text = "Camera access needed",
            color = AppColors.TextPrimary,
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "We use your camera only to capture a photo for grading. Nothing is recorded or shared.",
            color = AppColors.TextPrimary.copy(alpha = 0.7f),
            fontSize = 14.sp,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(
            onClick = onRequestAgain,
            colors = ButtonDefaults.buttonColors(containerColor = AppColors.Primary),
            shape = RoundedCornerShape(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Grant permission", color = AppColors.TextOnPrimary)
        }
    }
}

private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
    val buffer = image.planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

    val rotationDegrees = image.imageInfo.rotationDegrees
    val matrix = Matrix().apply {
        postRotate(rotationDegrees.toFloat())
        postScale(-1f, 1f)
    }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}