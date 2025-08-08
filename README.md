# MH Instruction Tuning - Model Performance Leaderboard

A Flask web application for tracking and visualizing model performance on psychological scales, with detailed comparison between model outputs and expert ratings.

## Features

- **Interactive Leaderboard**: Compare model performance across different scales with RMSE rankings
- **Detailed Scale Analysis**: Visualize model vs expert performance with interactive charts
- **Advanced Filtering**: Filter by temperature, top-p, system prompts, message prompts, and minimum runs
- **Prompt Content Display**: View full prompt content when filtering by specific prompts
- **Automatic Updates**: All filters update visualizations in real-time without page refreshes

## Setup Instructions

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. **Clone or download the project files**
   ```bash
   cd MH-Instruction-Tuning
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Setup**
   - You should receive a copy of the SQLite database file (`experiment_tracker.db`)
   - Place this file in the `instance/` directory
   - The database contains:
     - Scales and scale items with expert ratings
     - Experiments and results from model evaluations
     - Prompts (both system and message types)

## Running the Application

### Development Mode

1. **Start the Flask development server**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Open your web browser and go to: http://localhost:5000
   - The application will run in debug mode with automatic reloading

### Production Deployment

For production deployment, consider using a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Application Structure

```
MH-Instruction-Tuning/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── templates/
│   ├── leaderboard.html       # Main leaderboard page
│   └── scale_detail.html      # Detailed scale analysis page
├── instance/
│   └── experiment_tracker.db  # SQLite database (you'll receive this)
├── scripts/                   # Data processing scripts (excluded from repo)
├── data/                      # Data files (excluded from repo)
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## Using the Application

### Main Leaderboard (`/`)

- **View Rankings**: Models are ranked by RMSE (Root Mean Square Error) against expert ratings
- **Filter Options**:
  - **Temperature**: Model temperature settings (default: 0.0)
  - **Top-p**: Nucleus sampling parameter (default: 1.0)  
  - **System Prompt**: Filter by system prompt (includes "None" for null values)
  - **Message Prompt**: Filter by message prompt (includes "None" for null values)
  - **Min Runs**: Minimum number of complete scale runs required
- **Prompt Content**: Select a prompt to view its full content below the filters
- **Scale Details**: Click any row to view detailed analysis for that scale

### Scale Detail Pages (`/scale/<scale_id>`)

- **Interactive Charts**: Four separate charts showing different question groups
- **Model vs Expert**: Compare model performance against expert ratings
- **All Filters**: Same filtering options as leaderboard, plus additional top-p filtering
- **Statistics**: RMSE and coverage statistics for top-performing models

## Database Schema

The application uses the following main tables:

- **`scales`**: Psychological scales (e.g., SIRI-2)
- **`scale_items`**: Individual items within scales with expert ratings
- **`experiments`**: Experimental configurations and metadata
- **`results`**: Model responses and scores for each scale item
- **`prompts`**: System and message prompts used in experiments

## Key Metrics

- **RMSE**: Root Mean Square Error between model and expert ratings
- **Complete Runs**: Experiments with ≥80% of expected scale items completed
- **Coverage**: Percentage of scale items with valid model responses

## Dependencies

The project uses the following main dependencies (see `requirements.txt` for exact versions):

- **Flask**: Web framework
- **Flask-SQLAlchemy**: Database ORM
- **SQLAlchemy**: Database toolkit
- **numpy**: Numerical computing for RMSE calculations

## Troubleshooting

### Common Issues

1. **Database not found**
   - Ensure `experiment_tracker.db` is in the `instance/` directory
   - Check file permissions

2. **Import errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Ensure virtual environment is activated

3. **Port already in use**
   - The app runs on port 5000 by default
   - Kill existing processes: `lsof -ti:5000 | xargs kill -9` (macOS/Linux)
   - Or modify the port in `app.py`: `app.run(debug=True, port=8080)`

4. **No data showing**
   - Verify database contains data: Check that tables have records
   - Ensure scales are marked as `is_validated=True` and `is_public=True`

### Debug Mode

The application runs in debug mode by default, providing:
- Detailed error messages
- Automatic reloading on code changes
- Interactive debugger in the browser

## Development Notes

- **Automatic Filtering**: All filter changes trigger immediate updates
- **UUID Handling**: Prompt IDs are UUIDs, with special handling for "None" values
- **Color Coding**: Models are color-coded by family (GPT=blue, Claude=orange, Gemini=purple)
- **Expert Data**: Green bars represent expert consensus ratings

## File Organization

- **Excluded from Git**: `scripts/`, `data/`, `instance/`, and temporary files are excluded via `.gitignore`
- **Database Location**: SQLite database should be placed in `instance/experiment_tracker.db`
- **Templates**: HTML templates use modern JavaScript with automatic updates

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify database connectivity and location
3. Ensure all required dependencies are installed: `pip install -r requirements.txt`
4. Check that the database file has the expected schema and data

---

*Last updated: January 2025*