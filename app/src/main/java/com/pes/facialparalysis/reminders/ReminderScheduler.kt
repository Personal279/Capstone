package com.pes.facialparalysis.reminders

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import com.pes.facialparalysis.data.Reminder
import java.util.Calendar

/**
 * Schedules Android alarms for reminders the doctor/user explicitly created. Nothing in this
 * object decides *whether* a reminder should exist — it only turns an already-created [Reminder]
 * row into an OS-level alarm at the requested day(s)/time.
 */
object ReminderScheduler {

    private const val EXTRA_REMINDER_ID = "reminder_id"
    private const val EXTRA_TITLE = "title"
    private const val EXTRA_NOTES = "notes"
    private const val EXTRA_HOUR = "hour"
    private const val EXTRA_MINUTE = "minute"
    private const val EXTRA_DAYS_CSV = "days_csv"

    private fun pendingIntent(context: Context, reminder: Reminder): PendingIntent {
        val intent = Intent(context, ReminderAlarmReceiver::class.java).apply {
            putExtra(EXTRA_REMINDER_ID, reminder.id)
            putExtra(EXTRA_TITLE, reminder.title)
            putExtra(EXTRA_NOTES, reminder.notes)
            putExtra(EXTRA_HOUR, reminder.hour)
            putExtra(EXTRA_MINUTE, reminder.minute)
            putExtra(EXTRA_DAYS_CSV, reminder.daysOfWeekCsv)
        }
        val flags = PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        return PendingIntent.getBroadcast(context, reminder.id, intent, flags)
    }

    fun schedule(context: Context, reminder: Reminder) {
        val triggerAt = computeNextTriggerMillis(
            reminder.hour, reminder.minute, reminder.daysOfWeek(), reminder.oneTimeDateMillis
        ) ?: return // date/time already in the past and not recurring — nothing to schedule

        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val pi = pendingIntent(context, reminder)
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && !alarmManager.canScheduleExactAlarms()) {
                alarmManager.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi)
            } else {
                alarmManager.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi)
            }
        } catch (_: SecurityException) {
            alarmManager.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi)
        }
    }

    /** Re-arms a recurring reminder for its next occurrence; called by the receiver after it fires. */
    fun rescheduleRecurring(
        context: Context,
        reminderId: Int,
        title: String,
        notes: String,
        hour: Int,
        minute: Int,
        daysCsv: String
    ) {
        val days = daysCsv.split(",").mapNotNull { it.trim().toIntOrNull() }
        if (days.isEmpty()) return
        val triggerAt = computeNextTriggerMillis(hour, minute, days, 0L, skipToday = true) ?: return
        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(context, ReminderAlarmReceiver::class.java).apply {
            putExtra(EXTRA_REMINDER_ID, reminderId)
            putExtra(EXTRA_TITLE, title)
            putExtra(EXTRA_NOTES, notes)
            putExtra(EXTRA_HOUR, hour)
            putExtra(EXTRA_MINUTE, minute)
            putExtra(EXTRA_DAYS_CSV, daysCsv)
        }
        val pi = PendingIntent.getBroadcast(
            context, reminderId, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && !alarmManager.canScheduleExactAlarms()) {
                alarmManager.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi)
            } else {
                alarmManager.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi)
            }
        } catch (_: SecurityException) {
            alarmManager.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi)
        }
    }

    fun cancel(context: Context, reminderId: Int) {
        val alarmManager = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(context, ReminderAlarmReceiver::class.java)
        val pi = PendingIntent.getBroadcast(
            context, reminderId, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        alarmManager.cancel(pi)
    }

    /**
     * @param skipToday when true (used when re-arming right after firing), never returns a time
     * in the next few seconds even if today matches, so the same alarm can't double-fire.
     */
    private fun computeNextTriggerMillis(
        hour: Int,
        minute: Int,
        daysOfWeek: List<Int>,
        oneTimeDateMillis: Long,
        skipToday: Boolean = false
    ): Long? {
        val now = Calendar.getInstance()

        if (daysOfWeek.isEmpty()) {
            if (oneTimeDateMillis <= 0L) return null
            val cal = Calendar.getInstance().apply {
                timeInMillis = oneTimeDateMillis
                set(Calendar.HOUR_OF_DAY, hour)
                set(Calendar.MINUTE, minute)
                set(Calendar.SECOND, 0)
                set(Calendar.MILLISECOND, 0)
            }
            return if (cal.timeInMillis > now.timeInMillis) cal.timeInMillis else null
        }

        for (offset in 0..7) {
            val cal = Calendar.getInstance().apply {
                add(Calendar.DAY_OF_YEAR, offset)
                set(Calendar.HOUR_OF_DAY, hour)
                set(Calendar.MINUTE, minute)
                set(Calendar.SECOND, 0)
                set(Calendar.MILLISECOND, 0)
            }
            val dayMatches = daysOfWeek.contains(cal.get(Calendar.DAY_OF_WEEK))
            val isFutureEnough = if (offset == 0 && skipToday) false else cal.timeInMillis > now.timeInMillis
            if (dayMatches && isFutureEnough) return cal.timeInMillis
        }
        return null
    }
}
