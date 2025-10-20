from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0006_alter_processingsession_reference_document"),
    ]

    operations = [
        migrations.AddField(
            model_name="processingsession",
            name="processed_docx_with_diarization_path",
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
    ]
