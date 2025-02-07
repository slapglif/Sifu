import os
import shutil
from pathlib import Path
from scripts.visual_qa import visualizer, encode_image, resize_image
from loguru import logger

def test_visual_qa():
    """Test the visual QA functionality"""
    
    # Create test directory if it doesn't exist
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test image
    from PIL import Image, ImageDraw
    
    # Create a test image
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill='blue')
    draw.ellipse([75, 75, 125, 125], fill='red')
    
    # Save test image
    test_image_path = test_dir / "test_image.png"
    img.save(test_image_path)
    
    logger.info(f"Created test image at {test_image_path}")
    
    resized_path = None
    try:
        # Test image encoding
        logger.info("Testing image encoding...")
        encoded = encode_image(str(test_image_path))
        assert encoded, "Image encoding failed"
        logger.info("Image encoding successful")
        
        # Test image resizing
        logger.info("Testing image resizing...")
        resized_path = resize_image(str(test_image_path))
        assert os.path.exists(resized_path), "Image resizing failed"
        logger.info("Image resizing successful")
        
        # Test visual QA
        logger.info("Testing visual QA...")
        question = "What shapes can you see in this image?"
        result = visualizer.invoke({"image_path": str(test_image_path), "question": question})
        assert result, "Visual QA failed"
        logger.info(f"Visual QA result: {result}")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        try:
            if resized_path:
                resized_dir = Path(resized_path).parent
                if resized_dir.exists():
                    shutil.rmtree(resized_dir)
            if test_dir.exists():
                shutil.rmtree(test_dir)
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    test_visual_qa() 