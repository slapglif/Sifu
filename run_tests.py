import os
import sys
import pytest
from loguru import logger

def setup_logging():
    """Configure logging for tests"""
    log_file = "test_results.log"
    
    # Remove default handler
    logger.remove()
    
    # Add handlers for both file and console
    logger.add(log_file, rotation="1 day", retention="7 days", level="DEBUG")
    logger.add(sys.stderr, level="INFO")
    
    return log_file

def main():
    """Run the integration tests"""
    log_file = setup_logging()
    logger.info("Starting integration tests...")
    
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Configure pytest arguments
    pytest_args = [
        "integration_tests/test_visual_qa_system.py",
        "-v",                    # Verbose output
        "--capture=no",         # Show print statements
        "-s",                   # Show output
        "--tb=short",          # Shorter traceback format
        f"--log-file={log_file}",  # Log to file
        "--log-level=DEBUG",    # Log level
        "-n", "auto"           # Parallel execution
    ]
    
    try:
        # Run tests
        exit_code = pytest.main(pytest_args)
        
        if exit_code == 0:
            logger.info("All tests passed successfully!")
        else:
            logger.error(f"Tests failed with exit code: {exit_code}")
            
        logger.info(f"Test results have been saved to {log_file}")
        return exit_code
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 