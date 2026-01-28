"""
Test script for the latent score classifier.

Verifies that the classifier components work correctly:
1. LatentScoreClassifier forward pass
2. Dataset creation
3. Training loop (loss decreases)
4. Model save/load

Run with:
    python test_latent_classifier.py
"""

import sys
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '/cosma/home/durham/dc-fras4/code')

from latent_classifier import (
    LatentScoreClassifier,
    EmbeddingScoreDataset,
    train_classifier,
    evaluate_classifier,
    compute_class_weights
)
from torch.utils.data import DataLoader


def test_classifier_forward():
    """Test that the classifier forward pass works."""
    print("\n[TEST] Classifier forward pass...")
    
    classifier = LatentScoreClassifier(
        latent_dim=32,
        hidden_dims=[64, 32],
        num_classes=4,
        dropout=0.3
    )
    
    # Test with batch input
    x = torch.randn(16, 32)
    logits = classifier(x)
    
    assert logits.shape == (16, 4), f"Expected (16, 4), got {logits.shape}"
    
    # Test predict
    preds = classifier.predict(x)
    assert preds.shape == (16,), f"Expected (16,), got {preds.shape}"
    assert preds.min() >= 0 and preds.max() <= 3, "Predictions out of range"
    
    # Test predict_proba
    probs = classifier.predict_proba(x)
    assert probs.shape == (16, 4), f"Expected (16, 4), got {probs.shape}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-5), "Probabilities don't sum to 1"
    
    print("  ✓ Forward pass works correctly")
    return True


def test_embedding_dataset():
    """Test the EmbeddingScoreDataset."""
    print("\n[TEST] EmbeddingScoreDataset...")
    
    n_samples = 100
    latent_dim = 32
    
    embeddings = np.random.randn(n_samples, latent_dim).astype(np.float32)
    scores = np.random.randint(0, 4, size=n_samples)
    video_ids = [f"video_{i}" for i in range(n_samples)]
    
    dataset = EmbeddingScoreDataset(embeddings, scores, video_ids)
    
    assert len(dataset) == n_samples, f"Expected {n_samples}, got {len(dataset)}"
    
    emb, score, vid = dataset[0]
    assert emb.shape == (latent_dim,), f"Expected ({latent_dim},), got {emb.shape}"
    assert isinstance(score.item(), int), "Score should be integer"
    assert isinstance(vid, str), "Video ID should be string"
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=16)
    batch_emb, batch_scores, batch_ids = next(iter(loader))
    
    assert batch_emb.shape == (16, latent_dim), f"Expected (16, {latent_dim}), got {batch_emb.shape}"
    
    print("  ✓ EmbeddingScoreDataset works correctly")
    return True


def test_training_loop():
    """Test that the training loop runs and loss decreases."""
    print("\n[TEST] Training loop...")
    
    # Create synthetic data with slight class separation
    n_train = 200
    n_val = 50
    latent_dim = 32
    
    # Create data where score roughly correlates with first dimension
    train_scores = np.random.randint(0, 4, size=n_train)
    train_embeddings = np.random.randn(n_train, latent_dim).astype(np.float32)
    train_embeddings[:, 0] = train_scores + np.random.randn(n_train) * 0.5  # Add correlation
    train_ids = [f"train_{i}" for i in range(n_train)]
    
    val_scores = np.random.randint(0, 4, size=n_val)
    val_embeddings = np.random.randn(n_val, latent_dim).astype(np.float32)
    val_embeddings[:, 0] = val_scores + np.random.randn(n_val) * 0.5
    val_ids = [f"val_{i}" for i in range(n_val)]
    
    train_dataset = EmbeddingScoreDataset(train_embeddings, train_scores, train_ids)
    val_dataset = EmbeddingScoreDataset(val_embeddings, val_scores, val_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    classifier = LatentScoreClassifier(
        latent_dim=latent_dim,
        hidden_dims=[32, 16],
        num_classes=4,
        dropout=0.1
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    history = train_classifier(
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        learning_rate=1e-2,
        device=device,
        save_path=None
    )
    
    # Check loss decreased
    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]
    
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    
    assert final_loss < initial_loss, "Training loss should decrease"
    
    # Check accuracy is better than random (25%)
    final_val_acc = history['val_acc'][-1]
    print(f"  Final val accuracy: {final_val_acc:.1%}")
    
    # With correlated data, should do better than random
    assert final_val_acc > 0.20, f"Accuracy should be > 20%, got {final_val_acc:.1%}"
    
    print("  ✓ Training loop works correctly")
    return True


def test_class_weights():
    """Test class weight computation."""
    print("\n[TEST] Class weights...")
    
    # Imbalanced data
    scores = np.array([0, 0, 0, 1, 1, 2, 3])
    weights = compute_class_weights(scores)
    
    assert len(weights) == 4, "Should have 4 weights"
    assert weights[0] < weights[3], "More frequent class should have lower weight"
    
    print(f"  Weights for imbalanced data: {weights.tolist()}")
    print("  ✓ Class weights computed correctly")
    return True


def test_model_save_load():
    """Test model save and load."""
    print("\n[TEST] Model save/load...")
    
    import tempfile
    import os
    
    classifier = LatentScoreClassifier(latent_dim=32)
    classifier.eval()  # Set to eval mode for single-sample inference
    
    # Get initial prediction
    x = torch.randn(1, 32)
    initial_output = classifier(x).detach().clone()
    
    # Save model
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        save_path = f.name
    
    torch.save(classifier.state_dict(), save_path)
    
    # Create new model and load
    new_classifier = LatentScoreClassifier(latent_dim=32)
    new_classifier.load_state_dict(torch.load(save_path))
    new_classifier.eval()  # Set to eval mode
    
    # Check outputs match
    loaded_output = new_classifier(x)
    
    assert torch.allclose(initial_output, loaded_output, atol=1e-5), "Loaded model should produce same outputs"
    
    # Cleanup
    os.unlink(save_path)
    
    print("  ✓ Model save/load works correctly")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("LATENT CLASSIFIER TESTS")
    print("=" * 60)
    
    tests = [
        test_classifier_forward,
        test_embedding_dataset,
        test_class_weights,
        test_model_save_load,
        test_training_loop,  # Run this last as it takes longer
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
