package com.pes.facialparalysis.ui.theme.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.pes.facialparalysis.ml.LimeExplainer
import com.pes.facialparalysis.ui.theme.AppColors
import kotlin.math.hypot

/** Small teal uppercase label used at the top of every clinical screen ("GUIDED CAPTURE"). */
@Composable
fun EyebrowLabel(text: String, modifier: Modifier = Modifier, color: Color = AppColors.Primary) {
    Text(
        text = text.uppercase(),
        style = MaterialTheme.typography.labelMedium,
        color = color,
        modifier = modifier
    )
}

/** Large editorial-style display heading (Fraunces-style serif stand-in). */
@Composable
fun ScreenHeading(text: String, modifier: Modifier = Modifier, color: Color = AppColors.TextPrimary) {
    Text(
        text = text,
        style = MaterialTheme.typography.displayMedium,
        color = color,
        modifier = modifier
    )
}

/** Rounded, gently bordered card used for every clinical content block. */
@Composable
fun ClinicalCard(
    modifier: Modifier = Modifier,
    containerColor: Color = AppColors.Surface,
    contentPadding: androidx.compose.foundation.layout.PaddingValues = PaddingValues(18.dp),
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = containerColor),
        border = BorderStroke(1.dp, AppColors.Border),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(modifier = Modifier.padding(contentPadding), content = content)
    }
}

/** Full-width teal pill button used for every primary call-to-action. */
@Composable
fun PrimaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    showArrow: Boolean = true,
    leadingIcon: (@Composable () -> Unit)? = null
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier
            .fillMaxWidth()
            .height(56.dp),
        shape = RoundedCornerShape(28.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = AppColors.Primary,
            contentColor = AppColors.TextOnPrimary,
            disabledContainerColor = AppColors.Primary.copy(alpha = 0.4f)
        )
    ) {
        leadingIcon?.let { it(); Spacer(Modifier.width(8.dp)) }
        Text(text, style = MaterialTheme.typography.labelLarge)
        if (showArrow) {
            Spacer(Modifier.width(8.dp))
            Icon(Icons.Filled.ArrowForward, contentDescription = null, modifier = Modifier.size(18.dp))
        }
    }
}

/** Outlined/light counterpart to [PrimaryButton] for secondary actions. */
@Composable
fun SecondaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    leadingIcon: (@Composable () -> Unit)? = null
) {
    OutlinedButton(
        onClick = onClick,
        modifier = modifier.height(56.dp),
        shape = RoundedCornerShape(28.dp),
        border = BorderStroke(1.dp, AppColors.Border),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = AppColors.TextPrimary)
    ) {
        leadingIcon?.let { it(); Spacer(Modifier.width(8.dp)) }
        Text(text, style = MaterialTheme.typography.labelLarge)
    }
}

/** Label/value row used inside clinical metric cards (e.g. "Resting symmetry — 86%"). */
@Composable
fun ClinicalMetricRow(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    valueColor: Color = AppColors.TextPrimary
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = AppColors.TextSecondary)
        Text(value, style = MaterialTheme.typography.titleMedium, color = valueColor)
    }
}

/** A thin divider matching the clinical card's border color, for stacking [ClinicalMetricRow]s. */
@Composable
fun ClinicalDivider(modifier: Modifier = Modifier) {
    HorizontalDivider(modifier = modifier, thickness = 1.dp, color = AppColors.Border)
}

/** Small colored-dot + text status pill ("Tracking", "Position quality: good"). */
@Composable
fun StatusIndicator(text: String, modifier: Modifier = Modifier, color: Color = AppColors.Primary) {
    Row(modifier = modifier, verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(color)
        )
        Spacer(Modifier.width(6.dp))
        Text(text, style = MaterialTheme.typography.bodySmall, color = AppColors.TextSecondary)
    }
}

/** Numbered step row used on the guided-capture screen (e.g. "01  Neutral face"). */
@Composable
fun ProgressStepIndicator(steps: List<String>, activeIndex: Int, modifier: Modifier = Modifier) {
    Row(modifier = modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
        steps.forEachIndexed { index, label ->
            val active = index == activeIndex
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "%02d".format(index + 1),
                    style = MaterialTheme.typography.titleMedium,
                    color = if (active) AppColors.Primary else AppColors.TextMuted
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    text = label,
                    style = MaterialTheme.typography.bodySmall,
                    color = if (active) AppColors.TextPrimary else AppColors.TextSecondary,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

enum class StageState { PENDING, ACTIVE, DONE }

/** Row inside the AI-analysis checklist card, reflecting a real pipeline stage's progress. */
@Composable
fun StageChecklistRow(label: String, state: StageState, modifier: Modifier = Modifier) {
    Row(modifier = modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(22.dp)
                .clip(CircleShape)
                .background(if (state == StageState.PENDING) AppColors.SurfaceMuted else AppColors.Primary.copy(alpha = if (state == StageState.DONE) 1f else 0.15f)),
            contentAlignment = Alignment.Center
        ) {
            if (state == StageState.DONE) {
                Text("✓", color = AppColors.TextOnPrimary, style = MaterialTheme.typography.labelSmall)
            } else if (state == StageState.ACTIVE) {
                Box(Modifier.size(8.dp).clip(CircleShape).background(AppColors.Primary))
            }
        }
        Spacer(Modifier.width(12.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = if (state == StageState.PENDING) AppColors.TextSecondary else AppColors.TextPrimary
        )
    }
}

/** Selectable chip used for the XAI facial-region selector. */
@Composable
fun RegionChip(label: String, selected: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(18.dp))
            .background(if (selected) AppColors.Primary else AppColors.SurfaceMuted)
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 9.dp)
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = if (selected) AppColors.TextOnPrimary else AppColors.TextSecondary
        )
    }
}

/** One row of the recovery-dashboard timeline (a real [com.pes.facialparalysis.data.AssessmentRecord]). */
@Composable
fun TimelineItem(
    title: String,
    subtitle: String,
    isLatest: Boolean,
    onClick: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    ClinicalCard(
        modifier = modifier.then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 14.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(9.dp)
                    .clip(CircleShape)
                    .background(if (isLatest) AppColors.Primary else AppColors.Border)
            )
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.titleMedium, color = AppColors.TextPrimary)
                Text(subtitle, style = MaterialTheme.typography.bodySmall, color = AppColors.TextSecondary)
            }
        }
    }
}

/** Shared soft gradient backdrop for every clinical screen. */
@Composable
fun clinicalBackgroundBrush(): Brush = Brush.verticalGradient(
    colors = listOf(AppColors.BackgroundGradientTop, AppColors.BackgroundGradientBottom)
)

/**
 * Face/landmark visualization used on the splash, guided-capture, and landmark-mapping screens.
 *
 * When [landmarks] is null this renders a generic decorative glow silhouette (branding art only —
 * never presented as a real scan). When real MediaPipe landmarks are supplied, it renders the
 * actual detected points and a lightweight mesh connecting nearby points within each facial
 * region, so the visualization always reflects real geometry once one is available.
 */
@Composable
fun FaceMeshVisualization(
    modifier: Modifier = Modifier,
    landmarks: List<NormalizedLandmark>? = null,
    animated: Boolean = true,
    glowColor: Color = AppColors.AccentAqua
) {
    val transition = rememberInfiniteTransition(label = "mesh-pulse")
    val pulse by transition.animateFloat(
        initialValue = 0.55f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1400, easing = LinearEasing), RepeatMode.Reverse),
        label = "pulse"
    )
    val alpha = if (animated) pulse else 1f

    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        val silhouetteCenter = Offset(w / 2f, h / 2f)

        // Soft silhouette glow behind the mesh, present in both decorative and real-data modes.
        drawOval(
            brush = Brush.radialGradient(
                colors = listOf(glowColor.copy(alpha = 0.16f), Color.Transparent),
                center = silhouetteCenter,
                radius = w * 0.7f
            ),
            topLeft = Offset(w * 0.1f, h * 0.05f),
            size = androidx.compose.ui.geometry.Size(w * 0.8f, h * 0.9f)
        )

        if (landmarks.isNullOrEmpty()) {
            // Decorative fixed points approximating generic face proportions — brand art only.
            val decorativePoints = listOf(
                Offset(0.42f, 0.38f), Offset(0.58f, 0.38f), Offset(0.5f, 0.5f),
                Offset(0.4f, 0.62f), Offset(0.6f, 0.62f), Offset(0.5f, 0.68f),
                Offset(0.35f, 0.45f), Offset(0.65f, 0.45f)
            ).map { Offset(it.x * w, it.y * h) }
            for (i in decorativePoints.indices) {
                for (j in i + 1 until decorativePoints.size) {
                    val d = hypot(
                        (decorativePoints[i].x - decorativePoints[j].x),
                        (decorativePoints[i].y - decorativePoints[j].y)
                    )
                    if (d < w * 0.35f) {
                        drawLine(
                            color = glowColor.copy(alpha = 0.25f * alpha),
                            start = decorativePoints[i],
                            end = decorativePoints[j],
                            strokeWidth = 1.dp.toPx()
                        )
                    }
                }
            }
            decorativePoints.forEach {
                drawCircle(glowColor.copy(alpha = 0.9f * alpha), radius = 2.6.dp.toPx(), center = it)
            }
        } else {
            // Real detected geometry: sample a subset per region so the mesh reads clearly rather
            // than drawing all ~468 raw points.
            val regionIndices = LimeExplainer.regionLandmarkIndices
            val sampled = regionIndices.values.flatten().distinct()
                .mapNotNull { idx -> landmarks.getOrNull(idx) }
                .map { Offset(it.x() * w, it.y() * h) }
            for (i in sampled.indices) {
                for (j in i + 1 until sampled.size) {
                    val d = hypot((sampled[i].x - sampled[j].x), (sampled[i].y - sampled[j].y))
                    if (d < w * 0.22f) {
                        drawLine(
                            color = glowColor.copy(alpha = 0.3f),
                            start = sampled[i],
                            end = sampled[j],
                            strokeWidth = 1.dp.toPx()
                        )
                    }
                }
            }
            sampled.forEach {
                drawCircle(glowColor.copy(alpha = 0.95f * alpha), radius = 3.dp.toPx(), center = it)
            }
        }
    }
}

/** Simple back-chevron icon button matching the clinical card style used in headers. */
@Composable
fun ClinicalBackButton(onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .size(40.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(AppColors.Surface)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Icon(Icons.Filled.ArrowBack, contentDescription = "Back", tint = AppColors.TextPrimary, modifier = Modifier.size(18.dp))
    }
}
