from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0007_processingsession_processed_docx_with_diarization_path'),
    ]

    operations = [
        migrations.AddField(
            model_name='audiofile',
            name='diarization_status',
            field=models.CharField(
                default='pending',
                help_text='Tracks diarization lifecycle: pending, processing, completed, failed',
                max_length=32,
            ),
        ),
        migrations.AlterField(
            model_name='auditlog',
            name='object_id',
            field=models.CharField(max_length=255),
        ),
    ]
