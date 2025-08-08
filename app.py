"""
Complete Flask application with leaderboard and visualizations
"""

from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import text, func
import numpy as np
from collections import defaultdict
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///experiment_tracker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Custom UUID type for SQLite
class GUID(db.TypeDecorator):
    """Platform-independent GUID type using string."""
    impl = db.String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            if isinstance(value, uuid.UUID):
                return str(value)
            return value
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return uuid.UUID(value)
        return None


# Models (same as before with expert columns added)
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(GUID())
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = db.Column(GUID())


class Scale(db.Model):
    __tablename__ = 'scales'
    
    id = db.Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    scale_type = db.Column(db.String(50), nullable=False)
    version = db.Column(db.Integer, default=1)
    is_validated = db.Column(db.Boolean, default=False)
    is_public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(GUID())
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = db.Column(GUID())
    
    items = db.relationship('ScaleItem', backref='scale', lazy='dynamic', cascade='all, delete-orphan')


class ScaleItem(db.Model):
    __tablename__ = 'scale_items'
    
    id = db.Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    scale_id = db.Column(GUID(), db.ForeignKey('scales.id', ondelete='CASCADE'))
    prompt_id = db.Column(db.String(50), nullable=False)
    prompt_value = db.Column(db.Text, nullable=False)
    response_id = db.Column(db.String(50), nullable=False)
    response_value = db.Column(db.Text, nullable=False)
    position = db.Column(db.Integer, nullable=False)
    version = db.Column(db.Integer, default=1)
    expert_mean = db.Column(db.Float, nullable=True)
    expert_n = db.Column(db.Integer, nullable=True)
    expert_sd = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(GUID())
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = db.Column(GUID())


class Experiment(db.Model):
    __tablename__ = 'experiments'
    
    id = db.Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(GUID(), db.ForeignKey('users.id', ondelete='CASCADE'))
    scale_id = db.Column(GUID(), db.ForeignKey('scales.id'))
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    config = db.Column(db.JSON, nullable=False)
    status = db.Column(db.String(50), default='pending')
    task_id = db.Column(db.String(255))
    estimated_cost = db.Column(db.Float)
    actual_cost = db.Column(db.Float)
    progress = db.Column(db.Integer, default=0)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='experiments')
    scale = db.relationship('Scale', backref='experiments')


class Result(db.Model):
    __tablename__ = 'results'
    
    id = db.Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = db.Column(GUID(), db.ForeignKey('experiments.id', ondelete='CASCADE'))
    model = db.Column(db.String(100), nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    top_p = db.Column(db.Float, nullable=False)
    max_tokens = db.Column(db.Integer, nullable=False)
    system_prompt_id = db.Column(GUID(), db.ForeignKey('prompts.id'))
    message_prompt_id = db.Column(GUID(), db.ForeignKey('prompts.id'))
    scale_prompt_id = db.Column(db.String(50))
    scale_response_id = db.Column(db.String(50))
    repeat_number = db.Column(db.Integer)
    score = db.Column(db.Integer)
    reasoning = db.Column(db.Text)
    raw_response = db.Column(db.Text)
    response_time_ms = db.Column(db.Integer)
    token_usage = db.Column(db.JSON)
    status = db.Column(db.String(50), default='pending')
    error_type = db.Column(db.String(100))
    error_message = db.Column(db.Text)
    api_cost = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    experiment = db.relationship('Experiment', backref='results')

class Prompt(db.Model):
    __tablename__ = 'prompts'
    
    id = db.Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), unique=True, nullable=False)
    prompt_type = db.Column(db.String(50), nullable=False)  # 'system' or 'message'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(GUID())
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = db.Column(GUID())
    
    # Relationship to results
    message_results = db.relationship('Result', foreign_keys='Result.message_prompt_id', backref='message_prompt')
    system_results = db.relationship('Result', foreign_keys='Result.system_prompt_id', backref='system_prompt')


# Helper functions
def get_model_color(model_name):
    """Get color for model based on family"""
    model_lower = model_name.lower()
    if 'gpt' in model_lower or 'o1' in model_lower or 'o4' in model_lower:
        # GPT family - shades of blue
        if 'gpt-3.5' in model_lower:
            return '#BBDEFB'  # Light blue (was #E3F2FD)
        elif 'gpt-4o' in model_lower:
            return '#90CAF9'  # Medium blue (unchanged - already good)
        elif 'o1' in model_lower:
            return '#42A5F5'  # Darker blue (unchanged - already good)
        elif 'o4' in model_lower:
            return '#1E88E5'  # Deep blue (unchanged - already good)
        else:
            return '#2196F3'  # Default blue (unchanged - already good)
    elif 'claude' in model_lower:
        # Claude family - shades of orange
        if 'haiku' in model_lower:
            return '#FFE0B2'  # Light orange (was #FFF3E0)
        elif 'sonnet' in model_lower:
            return '#FFCC80'  # Medium orange (unchanged - already good)
        elif 'opus' in model_lower:
            return '#FF9800'  # Deep orange (unchanged - already good)
        else:
            return '#FFB74D'  # Default orange (unchanged - already good)
    elif 'gemini' in model_lower:
        # Gemini family - shades of purple
        if 'flash' in model_lower:
            return '#E1BEE7'  # Light purple (was #F3E5F5)
        elif 'pro' in model_lower:
            return '#9C27B0'  # Deep purple (unchanged - already good)
        else:
            return '#BA68C8'  # Default purple (unchanged - already good)
    else:
        return '#9E9E9E'  # Gray for unknown


def calculate_rmse(model_means, expert_means):
    """Calculate RMSE between model and expert means"""
    differences = []
    for key in model_means:
        if key in expert_means and expert_means[key] is not None:
            differences.append((model_means[key] - expert_means[key]) ** 2)
    
    if not differences:
        return None
    
    return np.sqrt(np.mean(differences))


# API Routes
@app.route('/')
def index():
    """Render the main leaderboard page"""
    return render_template('leaderboard.html')


@app.route('/api/leaderboard')
def get_leaderboard():
    """Get leaderboard data"""
    # Get filter parameters
    temperature = request.args.get('temperature', type=float)
    top_p = request.args.get('top_p', type=float)
    message_prompt_id = request.args.get('message_prompt_id', type=str)
    system_prompt_id = request.args.get('system_prompt_id', type=str)
    min_runs = request.args.get('min_runs', 10, type=int)
    
    # Get validated public scales
    scales = Scale.query.filter_by(is_validated=True, is_public=True).all()
    
    leaderboard_data = []
    
    for scale in scales:
        # Get expert means for this scale
        expert_items = ScaleItem.query.filter_by(scale_id=scale.id).filter(
            ScaleItem.expert_mean.isnot(None)
        ).all()
        
        expert_means = {}
        for item in expert_items:
            key = f"{item.prompt_id}_{item.response_id}"
            expert_means[key] = item.expert_mean
        
        # Build base query for filtering
        base_query = db.session.query(Result).join(Experiment).filter(
            Experiment.scale_id == scale.id
        )
        
        # Apply filters
        if temperature is not None:
            base_query = base_query.filter(Result.temperature == temperature)
        if top_p is not None:
            base_query = base_query.filter(Result.top_p == top_p)
        if message_prompt_id:
            # Filter by specific message prompt ID or null
            if message_prompt_id == 'null':
                base_query = base_query.filter(Result.message_prompt_id.is_(None))
            else:
                from uuid import UUID
                try:
                    prompt_uuid = UUID(message_prompt_id)
                    base_query = base_query.filter(Result.message_prompt_id == prompt_uuid)
                except ValueError:
                    # Invalid UUID format, skip filtering
                    pass
        if system_prompt_id:
            # Filter by specific system prompt ID or null
            if system_prompt_id == 'null':
                base_query = base_query.filter(Result.system_prompt_id.is_(None))
            else:
                from uuid import UUID
                try:
                    prompt_uuid = UUID(system_prompt_id)
                    base_query = base_query.filter(Result.system_prompt_id == prompt_uuid)
                except ValueError:
                    # Invalid UUID format, skip filtering
                    pass
        
        # Get results grouped by model and configuration
        results_query = base_query.with_entities(
            Result.model,
            Result.temperature,
            Result.top_p,
            Result.message_prompt_id,
            Result.system_prompt_id,
            Result.scale_prompt_id,
            Result.scale_response_id,
            Result.score,
            Result.repeat_number
        ).all()
        
        # First, determine what constitutes a "complete run" based on actual data
        # Find the most common set size across all configurations
        config_sizes = defaultdict(int)
        temp_configs = defaultdict(lambda: defaultdict(set))
        
        for result in results_query:
            config_key = f"{result.temperature}_{result.top_p}_{result.message_prompt_id}_{result.system_prompt_id}_{result.repeat_number}"
            item_key = f"{result.scale_prompt_id}_{result.scale_response_id}"
            temp_configs[result.model][config_key].add(item_key)
        
        # Find the most common configuration size (this is likely our "complete run" size)
        for model, configs in temp_configs.items():
            for config_key, items in configs.items():
                config_sizes[len(items)] += 1
        
        # Use the most frequent size, or fall back to the theoretical scale size
        if config_sizes:
            expected_items_count = max(config_sizes.keys())  # Use the largest observed set
        else:
            expected_items_count = ScaleItem.query.filter_by(scale_id=scale.id).count()
        
        # Group by model and configuration for real processing
        model_configs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> config -> repeat -> [results]
        
        for result in results_query:
            config_key = f"{result.temperature}_{result.top_p}_{result.message_prompt_id}_{result.system_prompt_id}"
            item_key = f"{result.scale_prompt_id}_{result.scale_response_id}"
            repeat = result.repeat_number or 0
            model_configs[result.model][config_key][repeat].append({
                'item_key': item_key,
                'score': result.score
            })
        
        # Process results by model
        model_data = defaultdict(lambda: {'scores': {}, 'total_runs': 0})
        
        for model, configs in model_configs.items():
            all_scores = defaultdict(list)
            total_repeats = 0
            
            for config_key, repeats in configs.items():
                for repeat_num, repeat_results in repeats.items():
                    items_in_repeat = set(r['item_key'] for r in repeat_results)
                    
                    # Count as a run if it has at least 80% of expected items
                    # This handles partial data while still ensuring meaningful results
                    if len(items_in_repeat) >= expected_items_count * 0.8:
                        total_repeats += 1
                        # Add scores from this repeat
                        for result in repeat_results:
                            all_scores[result['item_key']].append(result['score'])
            
            if total_repeats > 0:
                # Calculate average scores across all repeats
                avg_scores = {}
                for item_key, scores in all_scores.items():
                    if scores:
                        avg_scores[item_key] = sum(scores) / len(scores)
                
                model_data[model]['scores'] = avg_scores
                model_data[model]['total_runs'] = total_repeats
        
        # Calculate RMSE for each model
        for model, data in model_data.items():
            if data['total_runs'] >= min_runs:
                rmse = calculate_rmse(data['scores'], expert_means)
                if rmse is not None:
                    leaderboard_data.append({
                        'scale': scale.name,
                        'scale_id': str(scale.id),
                        'model': model,
                        'rmse': round(rmse, 3),
                        'n_runs': data['total_runs'],
                        'color': get_model_color(model)
                    })
    
    # Sort by RMSE (lower is better)
    leaderboard_data.sort(key=lambda x: x['rmse'])
    
    # Add rank
    for i, item in enumerate(leaderboard_data, 1):
        item['rank'] = i
    
    return jsonify(leaderboard_data)


@app.route('/api/parameters')
def get_available_parameters():
    """Get available parameter values for filters"""
    temperatures = db.session.query(Result.temperature).distinct().all()
    top_ps = db.session.query(Result.top_p).distinct().all()
    
    # Get prompts with their content and names
    prompts = db.session.query(Prompt.id, Prompt.name, Prompt.content, Prompt.prompt_type).all()
    prompt_options = []
    for prompt in prompts:
        # Create a meaningful display name
        if prompt.name:
            display_name = prompt.name
        else:
            # Try to extract a meaningful title from content (first sentence or line)
            content_lines = prompt.content.strip().split('\n')
            first_line = content_lines[0].strip()
            if len(first_line) > 80:
                display_name = first_line[:80] + "..."
            else:
                display_name = first_line or f"{prompt.content[:50]}..."
        
        prompt_options.append({
            'id': str(prompt.id),
            'name': display_name,
            'content_preview': prompt.content[:200] + "..." if len(prompt.content) > 200 else prompt.content,
            'type': prompt.prompt_type,
            'full_content': prompt.content  # Include full content for potential future use
        })
    
    prompt_options.sort(key=lambda x: x['name'])
    
    # Check if there are any results with null prompts
    has_null_message_prompts = db.session.query(Result).filter(Result.message_prompt_id.is_(None)).first() is not None
    has_null_system_prompts = db.session.query(Result).filter(Result.system_prompt_id.is_(None)).first() is not None
    
    return jsonify({
        'temperatures': sorted([t[0] for t in temperatures if t[0] is not None]),
        'top_ps': sorted([t[0] for t in top_ps if t[0] is not None]),
        'prompts': prompt_options,
        'has_null_message_prompts': has_null_message_prompts,
        'has_null_system_prompts': has_null_system_prompts
    })


@app.route('/scale/<scale_id>')
def scale_detail(scale_id):
    """Render detailed scale analysis page"""
    scale = Scale.query.get_or_404(scale_id)
    return render_template('scale_detail.html', scale=scale)


@app.route('/api/scale/<scale_id>/comparison')
def get_scale_comparison(scale_id):
    """Get comparison data for visualization"""
    temperature = request.args.get('temperature', 1.0, type=float)
    message_prompt_id = request.args.get('message_prompt_id', type=str)
    system_prompt_id = request.args.get('system_prompt_id', type=str)
    
    # Get scale and expert data
    scale = Scale.query.get_or_404(scale_id)
    expert_items = ScaleItem.query.filter_by(scale_id=scale_id).all()
    
    # Build expert data structure
    expert_data = {}
    for item in expert_items:
        item_key = f"{int(item.prompt_id)}{item.response_id.upper()}"
        expert_data[item_key] = {
            'mean': item.expert_mean,
            'sd': item.expert_sd
        }
    
    # Get model results
    query = db.session.query(
        Result.model,
        Result.scale_prompt_id,
        Result.scale_response_id,
        func.avg(Result.score).label('mean_score'),
        func.stddev(Result.score).label('std_score'),
        func.count(Result.id).label('n')
    ).join(Experiment).filter(
        Experiment.scale_id == scale_id,
        Result.temperature == temperature
    )
    
    if message_prompt_id:
        # Filter by specific message prompt ID or null
        if message_prompt_id == 'null':
            query = query.filter(Result.message_prompt_id.is_(None))
        else:
            from uuid import UUID
            try:
                prompt_uuid = UUID(message_prompt_id)
                query = query.filter(Result.message_prompt_id == prompt_uuid)
            except ValueError:
                # Invalid UUID format, skip filtering
                pass
    if system_prompt_id:
        # Filter by specific system prompt ID or null
        if system_prompt_id == 'null':
            query = query.filter(Result.system_prompt_id.is_(None))
        else:
            from uuid import UUID
            try:
                prompt_uuid = UUID(system_prompt_id)
                query = query.filter(Result.system_prompt_id == prompt_uuid)
            except ValueError:
                # Invalid UUID format, skip filtering
                pass
    
    results = query.group_by(
        Result.model,
        Result.scale_prompt_id,
        Result.scale_response_id
    ).all()
    
    # Process results
    model_data = defaultdict(list)
    for r in results:
        item_key = f"{int(r.scale_prompt_id)}{r.scale_response_id.upper()}"
        model_data[r.model].append({
            'item': item_key,
            'mean': r.mean_score,
            'std': r.std_score,
            'n': r.n
        })
    
    return jsonify({
        'expert': expert_data,
        'models': dict(model_data),
        'temperature': temperature,
        'message_prompt_id': message_prompt_id,
        'system_prompt_id': system_prompt_id
    })


@app.route('/api/scale/<scale_id>/plot')
def get_scale_plot_data(scale_id):
    """Get data formatted for bar plot visualization"""
    temperature = request.args.get('temperature', 1.0, type=float)
    top_p = request.args.get('top_p', type=float)
    message_prompt_id = request.args.get('message_prompt_id', type=str)
    system_prompt_id = request.args.get('system_prompt_id', type=str)
    
    # Get scale
    scale = Scale.query.get_or_404(scale_id)
    
    # Define item order (skip 14)
    items_order = []
    for i in range(1, 26):
        if i != 14:
            items_order.extend([f'{i}A', f'{i}B'])
    
    # Get expert data
    expert_items = ScaleItem.query.filter_by(scale_id=scale_id).all()
    expert_means = {}
    for item in expert_items:
        item_key = f"{int(item.prompt_id)}{item.response_id.upper()}"
        if item.expert_mean is not None:
            expert_means[item_key] = item.expert_mean
    
    # Get model results
    query = db.session.query(
        Result.model,
        Result.scale_prompt_id,
        Result.scale_response_id,
        func.avg(Result.score).label('mean_score')
    ).join(Experiment).filter(
        Experiment.scale_id == scale_id,
        Result.temperature == temperature
    )
    
    # Apply top_p filter if specified
    if top_p is not None:
        query = query.filter(Result.top_p == top_p)
    
    if message_prompt_id:
        # Filter by specific message prompt ID or null
        if message_prompt_id == 'null':
            query = query.filter(Result.message_prompt_id.is_(None))
        else:
            from uuid import UUID
            try:
                prompt_uuid = UUID(message_prompt_id)
                query = query.filter(Result.message_prompt_id == prompt_uuid)
            except ValueError:
                # Invalid UUID format, skip filtering
                pass
    if system_prompt_id:
        # Filter by specific system prompt ID or null
        if system_prompt_id == 'null':
            query = query.filter(Result.system_prompt_id.is_(None))
        else:
            from uuid import UUID
            try:
                prompt_uuid = UUID(system_prompt_id)
                query = query.filter(Result.system_prompt_id == prompt_uuid)
            except ValueError:
                # Invalid UUID format, skip filtering
                pass
    
    results = query.group_by(
        Result.model,
        Result.scale_prompt_id,
        Result.scale_response_id
    ).all()
    
    # Build data structure for plotting
    plot_data = defaultdict(dict)
    
    # Add expert data
    for item in items_order:
        if item in expert_means:
            plot_data[item]['Expert'] = expert_means[item]
    
    # Add model data
    for r in results:
        item_key = f"{int(r.scale_prompt_id)}{r.scale_response_id.upper()}"
        plot_data[item_key][r.model] = r.mean_score
    
    # Define model order
    model_order = [
        "Expert",
        "gpt-3.5-turbo-0125",
        "gpt-4o",
        "o1",
        "o4-mini",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro"
    ]
    
    # Get available models
    available_models = set(['Expert'])
    for item_data in plot_data.values():
        available_models.update(item_data.keys())
    
    # Filter and order models
    ordered_models = [m for m in model_order if m in available_models]
    
    # Create color map
    color_map = {model: get_model_color(model) for model in ordered_models}
    color_map['Expert'] = '#4CAF50'  # Green for expert
    
    # Format for frontend
    formatted_data = {
        'items': items_order,
        'models': ordered_models,
        'colors': color_map,
        'data': plot_data,
        'temperature': temperature,
        'message_prompt_id': message_prompt_id,
        'system_prompt_id': system_prompt_id,
        'scale_name': scale.name
    }
    
    return jsonify(formatted_data)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)