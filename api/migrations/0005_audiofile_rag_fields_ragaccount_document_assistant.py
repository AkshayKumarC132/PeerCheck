from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0004_ragaccount_vector_store_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="audiofile",
            name="rag_document_match_error",
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="audiofile",
            name="rag_document_match_status",
            field=models.CharField(blank=True, max_length=32, null=True),
        ),
        migrations.AddField(
            model_name="audiofile",
            name="rag_document_match_updated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="audiofile",
            name="rag_document_matches",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="ragaccount",
            name="document_match_assistant_id",
            field=models.CharField(blank=True, db_index=True, max_length=128, null=True),
        ),
    ]
