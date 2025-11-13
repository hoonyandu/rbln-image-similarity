import torch

def generate_clip_embeddings(model, processor, image):
    # image to tensor
    inputs = processor(images=image, return_tensors="pt")

    # generate embedding
    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"])
        embedding = outputs.last_hidden_state.cpu().numpy()

    # Remove batch dimension
    return embedding[0] # [1, 768]

def generate_qwen25vl_embeddings(model, processor, image):
    # image to tensor
    inputs = processor.image_processor(images=image, return_tensors="pt")

    # Extract features using model component
    with torch.no_grad():
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]

        outputs = model.visual(pixel_values, grid_thw)

        # Convert to numpy and apply mean pooling over variable sequence dimension
        embedding = outputs.cpu().numpy()
        embedding = embedding.mean(axis=0) # [3584, ]

    return embedding # [3584, ]

def generate_embeddings(model_type, model, processor, image):
    """
    Generate embeddings for a given image
    Args:
        model_type: model type
        model: model
        processor: processor
        image: image
    Returns:
        embedding: embedding
    """
    
    if model_type == "clip":
        embedding = generate_clip_embeddings(model, processor, image)
    elif model_type == "qwen25vl":
        embedding = generate_qwen25vl_embeddings(model, processor, image)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return embedding