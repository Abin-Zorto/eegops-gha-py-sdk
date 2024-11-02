# Create a feature metadata dictionary to specify dropped features
feature_metadata = {
    "dropped_features": ["Participant"]
}

# In your pipeline definition
create_rai_job = rai_components['constructor'](
    model_info=model_info,
    train_dataset=train_data,
    test_dataset=test_data,
    target_column_name='Remission',
    task_type='classification',
    categorical_column_names=['Participant'],
    classes=['Non-remission', 'Remission'],
    feature_metadata=feature_metadata  # Add this parameter instead of dropped_features
) 