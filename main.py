import logging
from rental_price_prediction.modeling.train import run_training_pipeline
from rental_price_prediction.plots import plot_model_comparison

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Colombo Rental Price Prediction Pipeline")
    
    try:
        results, best_model = run_training_pipeline()
        
        plot_model_comparison(results)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Best model R²: {results['Ensemble']['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()