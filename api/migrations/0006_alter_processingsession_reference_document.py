# Generated manually to allow ProcessingSession records without an attached reference document
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('api', '0005_audiofile_rag_fields_ragaccount_document_assistant'),
    ]

    operations = [
        migrations.AlterField(
            model_name='processingsession',
            name='reference_document',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to='api.referencedocument',
            ),
        ),
    ]
