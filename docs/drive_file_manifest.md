# Файлы для Google Drive

Положи в Google Drive папку:

```text
/content/drive/MyDrive/SPPR/
```

Внутри нее создай папку:

```text
/content/drive/MyDrive/SPPR/data/
```

И загрузи туда файлы из локальной машины:

## Обязательные файлы

1. `D:\Notebooks\sppr\laws.parquet`
   -> `/content/drive/MyDrive/SPPR/data/laws.parquet`

2. `D:\Notebooks\sppr\final_roles_punishments_v3.parquet`
   -> `/content/drive/MyDrive/SPPR/data/final_roles_punishments_v3.parquet`

3. `D:\Notebooks\sppr\cases_with_id.parquet`
   -> `/content/drive/MyDrive/SPPR/data/cases_with_id.parquet`

4. `D:\Notebooks\sppr\role_model.pkl`
   -> `/content/drive/MyDrive/SPPR/data/role_model.pkl`

5. `D:\Notebooks\sppr\vectorizer.pkl`
   -> `/content/drive/MyDrive/SPPR/data/vectorizer.pkl`

6. `D:\Notebooks\sppr\embeddings.pkl`
   -> `/content/drive/MyDrive/SPPR/data/embeddings.pkl`

7. `D:\Notebooks\sppr\faiss_index.bin`
   -> `/content/drive/MyDrive/SPPR/data/faiss_index.bin`

## Итоговая структура

```text
/content/drive/MyDrive/
  SPPR/
    data/
      laws.parquet
      final_roles_punishments_v3.parquet
      cases_with_id.parquet
      role_model.pkl
      vectorizer.pkl
      embeddings.pkl
      faiss_index.bin
    repo/
      SPPR-colab-backend/
```

## Что не нужно загружать в Drive для первого запуска

- промежуточные parquet
- ноутбуки из локального проекта
- локальный Gradio UI
- старые временные артефакты

## Репозиторий в Colab

Репозиторий можно не загружать в Drive вручную. Его проще клонировать прямо в Colab:

```python
!git clone https://github.com/Nephalem72/SPPR-colab-backend.git
```
