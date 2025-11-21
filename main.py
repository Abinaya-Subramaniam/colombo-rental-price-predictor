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
        
        best_r2 = results['Ensemble']['r2']
        logger.info("Pipeline completed successfully!")
        logger.info(f"Final Ensemble R²: {best_r2:.4f}")
        
        print("\n" + "="*50)
        print("FINAL MODEL SUMMARY")
        print("="*50)
        print("Selected: Huber + RANSAC Ensemble")
        print(f"Performance: R² = {best_r2:.4f}")
        print("Model saved to: models/best_rental_price_model.pkl")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()