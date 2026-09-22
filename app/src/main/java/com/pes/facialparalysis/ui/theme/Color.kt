package com.pes.facialparalysis.ui.theme

import androidx.compose.ui.graphics.Color

object AppColors {
    val Primary = Color(0xFF0F6E7A)
    val PrimaryDark = Color(0xFF0B4F58)
    val PrimaryLight = Color(0xFFE3F2F3)

    val Background = Color(0xFFF7F9FA)
    val BackgroundGradientTop = Color(0xFFF3F9FA)
    val BackgroundGradientBottom = Color(0xFFEAF3F3)
    val Surface = Color(0xFFFFFFFF)
    val SurfaceMuted = Color(0xFFEFF4F5)

    val TextPrimary = Color(0xFF1B2B2E)
    val TextSecondary = Color(0xFF5C6B6E)
    val TextMuted = Color(0xFFA9C0C0)
    val TextOnPrimary = Color(0xFFFFFFFF)

    val Border = Color(0xFFDCE5E6)

    /** Decorative visualization accents (face mesh glow, symmetry heatmap dots). Not used for
     * clinical grade semantics — [gradeColor] remains the source of truth there. */
    val AccentAqua = Color(0xFF2DD4C8)
    val AccentAquaSoft = Color(0xFFBEEBE6)
    val AccentAmber = Color(0xFFE0A03D)
    val AccentRose = Color(0xFFE0697D)

    val GradeNormal = Color(0xFF2E7D4F)
    val GradeMild = Color(0xFF8A7A1E)
    val GradeModerate = Color(0xFFB5651D)
    val GradeModSevere = Color(0xFFA84826)
    val GradeSevere = Color(0xFF9E3535)

    val Warning = Color(0xFFB5651D)
    val Error = Color(0xFF9E3535)

    fun gradeColor(grade: Int): Color = when (grade) {
        1 -> GradeNormal
        2 -> GradeMild
        3 -> GradeModerate
        4 -> GradeModSevere
        else -> GradeSevere
    }
}