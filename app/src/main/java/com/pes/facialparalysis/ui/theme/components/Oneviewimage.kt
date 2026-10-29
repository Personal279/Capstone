package com.pes.facialparalysis.ui.components

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.RemoveRedEye
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pes.facialparalysis.ui.theme.AppColors

/**
 * "One view" image: opens full once when tapped, then collapses to a blurred,
 * re-lockable placeholder. Local-only (state resets on recompose/navigation);
 * this is a viewing gate, not encryption — the file itself is unaffected.
 */
@Composable
fun OneViewImage(
    bitmap: Bitmap,
    modifier: Modifier = Modifier,
    height: androidx.compose.ui.unit.Dp = 220.dp
) {
    var revealed by remember { mutableStateOf(false) }
    var hasBeenViewed by remember { mutableStateOf(false) }
    val imageBitmap = remember(bitmap) { bitmap.asImageBitmap() }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(height)
            .clip(RoundedCornerShape(12.dp))
            .background(Color.Black)
            .clickable {
                revealed = !revealed
                if (revealed) hasBeenViewed = true
            },
        contentAlignment = Alignment.Center
    ) {
        Image(
            bitmap = imageBitmap,
            contentDescription = "Captured photo",
            contentScale = ContentScale.Crop,
            modifier = Modifier
                .fillMaxSize()
                .let { if (!revealed) it.blur(28.dp) else it }
        )

        if (!revealed) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = if (hasBeenViewed) Icons.Filled.VisibilityOff else Icons.Filled.RemoveRedEye,
                    contentDescription = null,
                    tint = Color.White
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = if (hasBeenViewed) "Tap to view again" else "Tap to view photo",
                    color = Color.White,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }
}