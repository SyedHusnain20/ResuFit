"""
skills_taxonomy.py — Master Skills Taxonomy
────────────────────────────────────────────
WHY A TAXONOMY FILE?
  A taxonomy is a structured classification system. Instead of a flat list of
  500 random skills, we group them by category. This gives us two things for
  free:
    1. Category-aware reporting — "You matched 4/6 Cloud skills"
    2. Easy maintenance — adding a new ML framework means editing one section

WHY NOT A DATABASE OR CSV?
  For a portfolio project, a Python dict is perfect. It's version-controlled,
  readable, and zero-dependency. If this were a real product at scale you'd
  move this to a DB so non-engineers could update it — but that's overkill here.

INTERVIEW TALKING POINT:
  "I chose a curated keyword taxonomy over NER because technical skills like
  FastAPI or TF-IDF aren't in any general-purpose NLP model's vocabulary.
  A curated list gives deterministic, explainable results — which matters
  when screening candidates."

STRUCTURE:
  Each key is a category name (shown in the API response).
  Each value is a list of skill strings exactly as they should be matched.
  Multi-word skills like "machine learning" are handled by the extractor.
"""

# ── Skills Taxonomy ───────────────────────────────────────────────────────────
# Format: { "Category Name": ["skill1", "skill2", ...] }

SKILLS_TAXONOMY: dict[str, list[str]] = {

    # ── Programming Languages ─────────────────────────────────────────────────
    "Programming Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "c",
        "go", "golang", "rust", "ruby", "php", "scala",
        "r", "matlab", "bash", "shell", "powershell", "perl", "dart",
        "haskell", "elixir", "clojure", "lua", "groovy",
    ],

    # ── Web Frameworks & Libraries ────────────────────────────────────────────
    "Web Frameworks": [
        "fastapi", "django", "flask", "express", "expressjs", "nestjs",
        "nextjs", "nuxtjs", "react", "reactjs", "angular", "vue", "vuejs",
        "svelte", "spring", "spring boot", "laravel", "rails", "ruby on rails",
        "asp.net", "blazor", "fastify", "hapi", "koa", "gin", "fiber",
        "tornado", "aiohttp", "starlette",
    ],

    # ── Databases ─────────────────────────────────────────────────────────────
    "Databases": [
        "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis",
        "elasticsearch", "cassandra", "dynamodb", "firestore", "firebase",
        "oracle", "sql server", "mssql", "mariadb", "cockroachdb",
        "neo4j", "couchdb", "influxdb", "timescaledb", "supabase",
        "planetscale", "fauna",
    ],

    # ── Cloud & DevOps ────────────────────────────────────────────────────────
    "Cloud & DevOps": [
        "aws", "amazon web services", "gcp", "google cloud", "azure",
        "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins",
        "github actions", "gitlab ci", "circleci", "travis ci",
        "heroku", "render", "vercel", "netlify", "digitalocean",
        "nginx", "apache", "linux", "unix", "ec2", "s3", "lambda",
        "cloudformation", "pulumi", "helm", "istio", "prometheus", "grafana",
    ],

    # ── Machine Learning & AI ─────────────────────────────────────────────────
    "Machine Learning & AI": [
        "machine learning", "deep learning", "neural networks",
        "natural language processing", "nlp", "computer vision",
        "reinforcement learning", "transfer learning",
        "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch",
        "hugging face", "transformers", "bert", "gpt", "llm",
        "xgboost", "lightgbm", "catboost", "random forest",
        "linear regression", "logistic regression", "svm",
        "support vector machine", "gradient boosting",
        "convolutional neural network", "cnn", "rnn", "lstm",
        "generative ai", "langchain", "llamaindex", "openai",
        "stable diffusion", "diffusion models",
    ],

    # ── Data Science & Analytics ──────────────────────────────────────────────
    "Data Science": [
        "pandas", "numpy", "matplotlib", "seaborn", "plotly",
        "jupyter", "scipy", "statsmodels", "data analysis",
        "data visualization", "data wrangling", "feature engineering",
        "etl", "data pipeline", "apache spark", "pyspark", "hadoop",
        "airflow", "dbt", "sql", "tableau", "power bi", "looker",
        "google analytics", "a/b testing", "hypothesis testing",
        "statistical analysis", "time series", "forecasting",
    ],

    # ── APIs & Integration ────────────────────────────────────────────────────
    "APIs & Integration": [
        "rest", "restful", "rest api", "graphql", "grpc", "websocket",
        "webhooks", "oauth", "jwt", "openapi", "swagger",
        "soap", "xml", "json", "protobuf", "api design",
        "microservices", "message queue", "rabbitmq", "kafka",
        "celery", "redis queue", "event driven",
    ],

    # ── Version Control & Collaboration ───────────────────────────────────────
    "Version Control": [
        "git", "github", "gitlab", "bitbucket", "svn",
        "code review", "pull request", "agile", "scrum", "kanban",
        "jira", "confluence", "notion", "linear",
    ],

    # ── Testing ───────────────────────────────────────────────────────────────
    "Testing": [
        "pytest", "unittest", "jest", "mocha", "chai", "cypress",
        "selenium", "playwright", "test driven development", "tdd",
        "bdd", "behavior driven development", "unit testing",
        "integration testing", "end to end testing", "e2e",
        "load testing", "performance testing",
    ],

    # ── Mobile Development ────────────────────────────────────────────────────
    "Mobile": [
        "react native", "flutter", "android", "ios", "swift",
        "kotlin", "xamarin", "ionic", "capacitor", "expo",
    ],

    # ── Security ─────────────────────────────────────────────────────────────
    "Security": [
        "cybersecurity", "penetration testing", "pen testing",
        "owasp", "encryption", "ssl", "tls", "https",
        "authentication", "authorization", "zero trust",
        "vulnerability assessment", "siem", "soc",
    ],

    # ── Soft Skills ───────────────────────────────────────────────────────────
    # These appear in JDs and resumes alike — worth matching
    "Soft Skills": [
        "communication", "teamwork", "leadership", "problem solving",
        "critical thinking", "time management", "collaboration",
        "mentoring", "project management", "stakeholder management",
        "presentation", "documentation", "technical writing",
    ],
}


# ── Derived Flat List ─────────────────────────────────────────────────────────
# A flat set of all skills — used for fast membership checks by the extractor.
# WHY A SET? O(1) lookup vs O(n) for a list. Matters when checking 500+ skills.
ALL_SKILLS: set[str] = {
    skill
    for skills in SKILLS_TAXONOMY.values()
    for skill in skills
}


# ── Reverse Lookup: skill → category ─────────────────────────────────────────
# Lets us quickly answer "what category is 'pytorch' in?" in O(1).
SKILL_TO_CATEGORY: dict[str, str] = {
    skill: category
    for category, skills in SKILLS_TAXONOMY.items()
    for skill in skills
}
