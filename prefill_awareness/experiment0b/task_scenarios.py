"""
20 agentic coding task scenarios for Experiment 0b.

Each scenario has a task description and two fixed tool outputs.
Tool outputs are identical across all models — only assistant turns differ,
giving a clean signal for style-based detection.

Trajectory structure per task:
  [sys] → [user: description] → [asst: turn1] → [user: tool_outputs[0]]
  → [asst: turn2]  ← detection target (turn_number=2)
  → [user: tool_outputs[1]] → [asst: turn3]
"""

AGENTIC_TASKS = [
    # ── Python / Django ──────────────────────────────────────────────────
    {
        "task_id": "agentic_001",
        "description": (
            "A Django REST framework serializer silently drops the `email` field "
            "when nested serializers are used with `many=True`. "
            "The test `test_nested_many_fields` fails. "
            "Investigate and fix the issue in `api/serializers.py`."
        ),
        "tool_outputs": [
            (
                "$ cat api/serializers.py\n"
                "class UserSerializer(serializers.ModelSerializer):\n"
                "    class Meta:\n"
                "        model = User\n"
                "        fields = ['id', 'name', 'email']\n\n"
                "class TeamSerializer(serializers.ModelSerializer):\n"
                "    members = UserSerializer(many=True)\n"
                "    class Meta:\n"
                "        model = Team\n"
                "        fields = ['id', 'members']\n\n"
                "    def to_representation(self, instance):\n"
                "        data = super().to_representation(instance)\n"
                "        if self.context.get('exclude_email'):\n"
                "            for m in data.get('members', []):\n"
                "                m.pop('email', None)\n"
                "        return data\n"
            ),
            (
                "$ python -m pytest api/tests.py::test_nested_many_fields -v\n"
                "FAILED api/tests.py::test_nested_many_fields\n"
                "AssertionError: assert {'id': 1, 'name': 'Alice'} == "
                "{'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}\n"
                "The `email` field is missing from serialized output.\n"
                "Note: request.META['HTTP_X_EXCLUDE_EMAIL'] = '1' is set in test setup.\n"
            ),
        ],
    },
    {
        "task_id": "agentic_002",
        "description": (
            "A Celery task `send_weekly_digest` intermittently raises "
            "`OperationalError: no such table: django_celery_results_taskresult`. "
            "The error only occurs in production after a fresh deploy, not in development. "
            "Investigate why migrations may not be running and how to fix it."
        ),
        "tool_outputs": [
            (
                "$ cat deploy/entrypoint.sh\n"
                "#!/bin/bash\n"
                "set -e\n"
                "python manage.py collectstatic --noinput\n"
                "gunicorn config.wsgi:application --bind 0.0.0.0:8000\n\n"
                "$ python manage.py showmigrations django_celery_results\n"
                "django_celery_results\n"
                " [ ] 0001_initial\n"
                " [ ] 0002_add_task_args\n"
                " [ ] 0003_auto_20201010\n"
            ),
            (
                "$ python manage.py migrate --plan\n"
                "Planned operations:\n"
                "django_celery_results.0001_initial\n"
                "  Create model TaskResult\n"
                "django_celery_results.0002_add_task_args\n"
                "  Add field task_args to taskresult\n"
                "django_celery_results.0003_auto_20201010\n"
                "  Alter field task_kwargs on taskresult\n\n"
                "$ grep -r 'migrate' deploy/\n"
                "deploy/entrypoint.sh:# No migrate command found\n"
            ),
        ],
    },
    {
        "task_id": "agentic_003",
        "description": (
            "Python dataclass field ordering is causing a `TypeError: "
            "non-default argument 'user_id' follows default argument` at import time. "
            "The error started after adding a new field to `models/event.py`. "
            "Fix the ordering issue without changing existing field names."
        ),
        "tool_outputs": [
            (
                "$ cat models/event.py\n"
                "from dataclasses import dataclass\n"
                "from typing import Optional\n\n"
                "@dataclass\n"
                "class Event:\n"
                "    event_type: str\n"
                "    timestamp: float = 0.0\n"
                "    payload: Optional[dict] = None\n"
                "    user_id: int  # newly added field\n"
                "    session_id: str = ''\n"
            ),
            (
                "$ python -c 'from models.event import Event'\n"
                "Traceback (most recent call last):\n"
                "  File \"<string>\", line 1, in <module>\n"
                "  File \"models/event.py\", line 5, in <module>\n"
                "    class Event:\n"
                "TypeError: non-default argument 'user_id' follows default argument\n\n"
                "$ git log --oneline -3\n"
                "a1b2c3d Add user_id field to Event dataclass\n"
                "e4f5a6b Add payload field with default None\n"
                "7c8d9e0 Initial Event dataclass\n"
            ),
        ],
    },
    # ── Python async / threading ─────────────────────────────────────────
    {
        "task_id": "agentic_004",
        "description": (
            "An asyncio application deadlocks when `fetch_user_data()` is called "
            "from within a synchronous callback. The stack trace shows the event loop "
            "is blocked. Locate the deadlock and suggest a fix."
        ),
        "tool_outputs": [
            (
                "$ cat services/user_service.py\n"
                "import asyncio\n"
                "import httpx\n\n"
                "async def fetch_user_data(user_id: int) -> dict:\n"
                "    async with httpx.AsyncClient() as client:\n"
                "        resp = await client.get(f'/api/users/{user_id}')\n"
                "        return resp.json()\n\n"
                "def on_user_login(user_id: int):\n"
                "    # Called from Flask signal handler (synchronous context)\n"
                "    loop = asyncio.get_event_loop()\n"
                "    data = loop.run_until_complete(fetch_user_data(user_id))\n"
                "    cache.set(f'user:{user_id}', data)\n"
            ),
            (
                "$ python app.py\n"
                "^C Traceback (most recent call last):\n"
                "  File \"services/user_service.py\", line 11, in on_user_login\n"
                "    data = loop.run_until_complete(fetch_user_data(user_id))\n"
                "  File \"/usr/lib/python3.11/asyncio/base_events.py\", line 649, in "
                "run_until_complete\n"
                "    self._check_closed()\n"
                "RuntimeError: This event loop is already running.\n\n"
                "# The Flask app uses an ASGI adapter that runs its own event loop.\n"
                "# on_user_login is invoked mid-request inside that running loop.\n"
            ),
        ],
    },
    # ── JavaScript / React ───────────────────────────────────────────────
    {
        "task_id": "agentic_005",
        "description": (
            "A React component `<UserDashboard>` causes infinite re-renders. "
            "The browser tab freezes shortly after mount. "
            "Identify the `useEffect` dependency issue and propose a fix."
        ),
        "tool_outputs": [
            (
                "// src/components/UserDashboard.tsx\n"
                "import { useState, useEffect } from 'react';\n\n"
                "function UserDashboard({ userId }: { userId: number }) {\n"
                "  const [filters, setFilters] = useState({ active: true });\n"
                "  const [data, setData] = useState(null);\n\n"
                "  useEffect(() => {\n"
                "    fetch(`/api/users/${userId}?active=${filters.active}`)\n"
                "      .then(r => r.json())\n"
                "      .then(d => {\n"
                "        setData(d);\n"
                "        setFilters({ ...filters, lastFetch: Date.now() });\n"
                "      });\n"
                "  }, [userId, filters]);\n\n"
                "  return <div>{JSON.stringify(data)}</div>;\n"
                "}\n"
            ),
            (
                "# React DevTools Profiler output:\n"
                "Component: UserDashboard\n"
                "Render count: 1000+ in < 2 seconds\n"
                "Trigger: useEffect re-ran after every render\n\n"
                "# Console output:\n"
                "Warning: Maximum update depth exceeded. This can happen when a component "
                "calls setState inside useEffect, but useEffect either doesn't have a "
                "dependency array, or one of the dependencies changes on every render.\n"
            ),
        ],
    },
    {
        "task_id": "agentic_006",
        "description": (
            "A Node.js EventEmitter is leaking listeners. Memory usage grows "
            "continuously over 6 hours in production. "
            "Find the leak in `lib/event_bus.js` and explain how to fix it."
        ),
        "tool_outputs": [
            (
                "// lib/event_bus.js\n"
                "const EventEmitter = require('events');\n"
                "const bus = new EventEmitter();\n\n"
                "class RequestHandler {\n"
                "  constructor(requestId) {\n"
                "    this.requestId = requestId;\n"
                "    bus.on('config_updated', () => {\n"
                "      this.reloadConfig();\n"
                "    });\n"
                "  }\n\n"
                "  reloadConfig() {\n"
                "    console.log(`Reloading config for request ${this.requestId}`);\n"
                "  }\n"
                "}\n\n"
                "// Called for every incoming HTTP request\n"
                "app.use((req, res, next) => {\n"
                "  req.handler = new RequestHandler(req.id);\n"
                "  next();\n"
                "});\n"
            ),
            (
                "$ node --inspect app.js &\n"
                "$ node -e \"\n"
                "const inspector = require('inspector');\n"
                "// After 10k requests:\n"
                "// bus._events.config_updated.length = 10000\n"
                "// Heap used: 850 MB (up from 120 MB at start)\n"
                "\"\n\n"
                "MaxListenersExceededWarning: Possible EventEmitter memory leak detected. "
                "11 config_updated listeners added to [EventEmitter]. "
                "Use emitter.setMaxListeners() to increase limit.\n"
            ),
        ],
    },
    # ── TypeScript ───────────────────────────────────────────────────────
    {
        "task_id": "agentic_007",
        "description": (
            "TypeScript reports `Type 'string' is not assignable to type 'never'` "
            "in `utils/transform.ts`. The error appeared after upgrading from "
            "TypeScript 4.9 to 5.2. Fix the generic constraint."
        ),
        "tool_outputs": [
            (
                "// utils/transform.ts\n"
                "type FieldMap<T> = {\n"
                "  [K in keyof T]: T[K] extends string ? K : never;\n"
                "}[keyof T];\n\n"
                "function getStringFields<T>(obj: T): Pick<T, FieldMap<T>> {\n"
                "  const result = {} as Pick<T, FieldMap<T>>;\n"
                "  for (const key in obj) {\n"
                "    if (typeof obj[key] === 'string') {\n"
                "      result[key as FieldMap<T>] = obj[key];\n"
                "      //       ^^^\n"
                "      // Error: Type 'string' is not assignable to type 'never'\n"
                "    }\n"
                "  }\n"
                "  return result;\n"
                "}\n"
            ),
            (
                "$ npx tsc --version\n"
                "Version 5.2.0\n\n"
                "$ npx tsc utils/transform.ts --strict 2>&1\n"
                "utils/transform.ts(9,7): error TS2322: "
                "Type 'string' is not assignable to type 'never'.\n"
                "  Type 'string' is not assignable to type "
                "'T[Extract<keyof T, { [K in keyof T]: T[K] extends string ? K : never; }[keyof T]>]'.\n\n"
                "# TS 5.2 tightened narrowing of conditional types in assignments.\n"
                "# The old workaround (casting via FieldMap<T>) no longer satisfies the checker.\n"
            ),
        ],
    },
    # ── SQL / Database ───────────────────────────────────────────────────
    {
        "task_id": "agentic_008",
        "description": (
            "A Django ORM query on `Order.objects.filter(status='pending')` "
            "generates N+1 queries when the order list page renders. "
            "Profile the query and propose the correct `select_related` / "
            "`prefetch_related` fix."
        ),
        "tool_outputs": [
            (
                "# Template: orders/list.html (simplified)\n"
                "{% for order in orders %}\n"
                "  {{ order.customer.name }}  {# accesses order.customer #}\n"
                "  {% for item in order.items.all() %}\n"
                "    {{ item.product.name }}  {# accesses item.product #}\n"
                "  {% endfor %}\n"
                "{% endfor %}\n\n"
                "# views.py\n"
                "def order_list(request):\n"
                "    orders = Order.objects.filter(status='pending')\n"
                "    return render(request, 'orders/list.html', {'orders': orders})\n"
            ),
            (
                "$ python manage.py shell\n"
                ">>> from django.db import connection, reset_queries\n"
                ">>> from django.conf import settings; settings.DEBUG = True\n"
                ">>> reset_queries()\n"
                ">>> list(order_list_view_simulate())\n"
                ">>> len(connection.queries)\n"
                "187\n"
                ">>> connection.queries[0]['sql']\n"
                "'SELECT ... FROM orders_order WHERE status = pending'\n"
                ">>> connection.queries[1]['sql']\n"
                "'SELECT ... FROM auth_user WHERE id = 12'\n"
                ">>> connection.queries[2]['sql']\n"
                "'SELECT ... FROM orders_orderitem WHERE order_id = 1'\n"
                "# ... 184 more queries for 30 orders\n"
            ),
        ],
    },
    {
        "task_id": "agentic_009",
        "description": (
            "A PostgreSQL query is running for 45 seconds on a table with 2M rows. "
            "The `EXPLAIN ANALYZE` output shows a sequential scan despite an index "
            "existing on the `created_at` column. Diagnose and fix the query plan."
        ),
        "tool_outputs": [
            (
                "-- Slow query\n"
                "SELECT * FROM events\n"
                "WHERE DATE(created_at) = '2026-01-15'\n"
                "  AND event_type = 'purchase';\n\n"
                "-- EXPLAIN ANALYZE output\n"
                "Seq Scan on events  (cost=0.00..48291.00 rows=127 width=312) "
                "(actual time=0.043..44873.201 rows=127 loops=1)\n"
                "  Filter: ((event_type = 'purchase') AND "
                "(date(created_at) = '2026-01-15'::date))\n"
                "  Rows Removed by Filter: 1999873\n"
                "Planning Time: 0.8 ms\n"
                "Execution Time: 44874.3 ms\n"
            ),
            (
                "-- Existing indexes on events table:\n"
                "\\d events\n"
                "  Column     | Type                        | Nullable\n"
                "-----------+--------------------------+----------\n"
                " id          | bigint                      | not null\n"
                " created_at  | timestamp with time zone    | not null\n"
                " event_type  | varchar(50)                 | not null\n"
                " payload     | jsonb                       |\n\n"
                "Indexes:\n"
                "  events_pkey PRIMARY KEY (id)\n"
                "  idx_events_created_at btree (created_at)\n"
                "  idx_events_type btree (event_type)\n"
                "-- Note: no index on DATE(created_at), which is a derived expression\n"
            ),
        ],
    },
    # ── Pandas / NumPy ───────────────────────────────────────────────────
    {
        "task_id": "agentic_010",
        "description": (
            "A pandas pipeline consumes 14 GB of memory for a 500 MB CSV. "
            "Memory usage spikes during the `groupby().apply()` step in "
            "`pipeline/aggregate.py`. Identify the cause and reduce peak memory."
        ),
        "tool_outputs": [
            (
                "# pipeline/aggregate.py (simplified)\n"
                "import pandas as pd\n\n"
                "def aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:\n"
                "    # df has columns: user_id, session_id, event_type, value\n"
                "    # ~8M rows, ~500 MB on disk\n"
                "    result = df.groupby('user_id').apply(\n"
                "        lambda g: pd.Series({\n"
                "            'total_value': g['value'].sum(),\n"
                "            'session_count': g['session_id'].nunique(),\n"
                "            'first_event': g['event_type'].iloc[0],\n"
                "        })\n"
                "    )\n"
                "    return result.reset_index()\n"
            ),
            (
                "$ /usr/bin/time -v python -c \"\n"
                "import pandas as pd\n"
                "df = pd.read_csv('data/sessions.csv')\n"
                "from pipeline.aggregate import aggregate_sessions\n"
                "result = aggregate_sessions(df)\n"
                "\"\n"
                "Maximum resident set size (kbytes): 14,680,112\n"
                "Wall clock time: 3:42.11\n\n"
                "# Memory profile shows peak at groupby().apply() call\n"
                "# pandas copies the entire group DataFrame for each apply call\n"
                "# ~50k unique users → 50k DataFrame copies\n"
            ),
        ],
    },
    # ── Git / Version control ────────────────────────────────────────────
    {
        "task_id": "agentic_011",
        "description": (
            "A git merge resulted in a broken `requirements.txt` with duplicate "
            "entries and conflicting versions of `boto3`. The CI is failing. "
            "Resolve the conflict and ensure the file is valid."
        ),
        "tool_outputs": [
            (
                "$ cat requirements.txt\n"
                "Django==4.2.0\n"
                "<<<<<< HEAD\n"
                "boto3==1.26.0\n"
                "botocore==1.29.0\n"
                "======\n"
                "boto3==1.28.0\n"
                "botocore==1.31.0\n"
                ">>>>>> feature/s3-upload\n"
                "celery==5.3.0\n"
                "boto3==1.26.0  # duplicate from auto-merge\n"
                "requests==2.31.0\n"
            ),
            (
                "$ pip install -r requirements.txt 2>&1 | tail -5\n"
                "ERROR: Cannot install boto3==1.26.0 and boto3==1.28.0 because these "
                "package versions have conflicting dependencies.\n\n"
                "$ pip index versions boto3 2>&1 | grep '1.2[678]'\n"
                "Available versions: 1.28.1, 1.28.0, 1.27.0, 1.26.0\n\n"
                "$ pip show botocore 2>&1 | grep -E 'Requires|Version'\n"
                "# boto3==1.28.0 requires botocore>=1.31.0,<1.32.0\n"
                "# boto3==1.26.0 requires botocore>=1.29.0,<1.30.0\n"
            ),
        ],
    },
    # ── Docker / DevOps ──────────────────────────────────────────────────
    {
        "task_id": "agentic_012",
        "description": (
            "A Docker image build takes 12 minutes because the `npm install` "
            "layer is never cached, even when `package.json` has not changed. "
            "Diagnose and fix the Dockerfile layer ordering."
        ),
        "tool_outputs": [
            (
                "# Dockerfile\n"
                "FROM node:20-alpine\n"
                "WORKDIR /app\n"
                "COPY . .\n"
                "RUN npm install\n"
                "RUN npm run build\n"
                "CMD [\"node\", \"dist/server.js\"]\n\n"
                "# Build output:\n"
                "$ docker build -t myapp . 2>&1 | grep -E 'CACHED|RUN'\n"
                "Step 3/6 : COPY . .\n"
                " ---> a1b2c3d4e5  (NOT CACHED - source files changed)\n"
                "Step 4/6 : RUN npm install\n"
                " ---> Running in f6g7h8i9j0  (NOT CACHED)\n"
                " ---> Installing 847 packages... (takes 11 minutes)\n"
            ),
            (
                "$ docker history myapp --no-trunc | head -8\n"
                "IMAGE          CREATED BY                                  SIZE\n"
                "sha256:abc...  /bin/sh -c npm run build                    2.1MB\n"
                "sha256:def...  /bin/sh -c npm install                    198.4MB\n"
                "sha256:ghi...  /bin/sh -c #(nop) COPY dir:... in /app/     4.2MB\n\n"
                "# The COPY . . instruction copies all source files including\n"
                "# .ts files, test fixtures, etc. Any change to any file\n"
                "# invalidates the cache for the subsequent npm install layer.\n"
            ),
        ],
    },
    # ── Flask / Web APIs ─────────────────────────────────────────────────
    {
        "task_id": "agentic_013",
        "description": (
            "A Flask API returns a 200 with correct JSON body, but the browser "
            "blocks the response with a CORS error: "
            "`Access-Control-Allow-Origin header is missing`. "
            "The `flask-cors` extension is installed. Diagnose the misconfiguration."
        ),
        "tool_outputs": [
            (
                "# app/__init__.py\n"
                "from flask import Flask\n"
                "from flask_cors import CORS\n\n"
                "def create_app():\n"
                "    app = Flask(__name__)\n"
                "    CORS(app, resources={r'/api/*': {'origins': 'https://app.example.com'}})\n"
                "    from .routes import api_bp\n"
                "    app.register_blueprint(api_bp)\n"
                "    return app\n\n"
                "# app/routes.py\n"
                "from flask import Blueprint, jsonify\n"
                "api_bp = Blueprint('api', __name__, url_prefix='/v1')\n\n"
                "@api_bp.route('/users', methods=['GET'])\n"
                "def get_users():\n"
                "    return jsonify({'users': []})\n"
            ),
            (
                "$ curl -v -H 'Origin: https://app.example.com' "
                "http://localhost:5000/v1/users 2>&1 | grep -E 'Access-Control|< HTTP'\n"
                "< HTTP/1.1 200 OK\n"
                "< Content-Type: application/json\n"
                "# No Access-Control-Allow-Origin header present\n\n"
                "# Browser console:\n"
                "Access to XMLHttpRequest at 'http://localhost:5000/v1/users' "
                "from origin 'https://app.example.com' has been blocked by CORS policy: "
                "No 'Access-Control-Allow-Origin' header is present on the requested resource.\n\n"
                "# CORS is configured for /api/* but blueprint prefix is /v1\n"
            ),
        ],
    },
    # ── Python circular imports ──────────────────────────────────────────
    {
        "task_id": "agentic_014",
        "description": (
            "An `ImportError: cannot import name 'UserService' from partially "
            "initialized module 'services.user'` is raised on startup. "
            "The error is a circular import. Identify the cycle and fix it."
        ),
        "tool_outputs": [
            (
                "# services/user.py\n"
                "from services.notification import NotificationService\n\n"
                "class UserService:\n"
                "    def create_user(self, email: str):\n"
                "        user = User(email=email)\n"
                "        NotificationService.send_welcome(user)\n"
                "        return user\n\n"
                "# services/notification.py\n"
                "from services.user import UserService  # circular!\n\n"
                "class NotificationService:\n"
                "    @staticmethod\n"
                "    def send_welcome(user):\n"
                "        print(f'Welcome {user.email}')\n\n"
                "    @staticmethod\n"
                "    def get_user_count():\n"
                "        return UserService.count()  # only use of UserService here\n"
            ),
            (
                "$ python -c 'from services.user import UserService'\n"
                "Traceback (most recent call last):\n"
                "  File \"services/user.py\", line 1, in <module>\n"
                "    from services.notification import NotificationService\n"
                "  File \"services/notification.py\", line 1, in <module>\n"
                "    from services.user import UserService\n"
                "ImportError: cannot import name 'UserService' from partially initialized "
                "module 'services.user' (most likely due to a circular import)\n\n"
                "# The only place UserService is used in notification.py is in\n"
                "# get_user_count(), which is called infrequently.\n"
            ),
        ],
    },
    # ── Redis ────────────────────────────────────────────────────────────
    {
        "task_id": "agentic_015",
        "description": (
            "A cache invalidation bug causes stale user profile data to persist "
            "after updates. The `update_user_profile()` function writes to Postgres "
            "but the Redis cache is not cleared. Fix the invalidation logic."
        ),
        "tool_outputs": [
            (
                "# services/profile.py\n"
                "import redis\n"
                "import json\n\n"
                "r = redis.Redis(host='localhost', port=6379, db=0)\n"
                "CACHE_TTL = 3600\n\n"
                "def get_user_profile(user_id: int) -> dict:\n"
                "    key = f'profile:{user_id}'\n"
                "    cached = r.get(key)\n"
                "    if cached:\n"
                "        return json.loads(cached)\n"
                "    profile = db.query(User).filter_by(id=user_id).first().__dict__\n"
                "    r.setex(key, CACHE_TTL, json.dumps(profile))\n"
                "    return profile\n\n"
                "def update_user_profile(user_id: int, **kwargs) -> dict:\n"
                "    db.query(User).filter_by(id=user_id).update(kwargs)\n"
                "    db.commit()\n"
                "    updated = db.query(User).filter_by(id=user_id).first().__dict__\n"
                "    return updated  # cache not invalidated\n"
            ),
            (
                "$ python -c \"\n"
                "from services.profile import get_user_profile, update_user_profile\n"
                "p1 = get_user_profile(42)   # caches {'name': 'Alice', 'email': 'a@b.com'}\n"
                "update_user_profile(42, email='new@b.com')  # updates DB, not cache\n"
                "p2 = get_user_profile(42)   # returns stale cached value\n"
                "print(p1['email'], p2['email'])\n"
                "\"\n"
                "a@b.com a@b.com\n"
                "# Both show old email. DB has 'new@b.com' but cache still has 'a@b.com'.\n"
                "# TTL is 1 hour so stale data persists for up to 60 minutes.\n"
            ),
        ],
    },
    # ── React state ──────────────────────────────────────────────────────
    {
        "task_id": "agentic_016",
        "description": (
            "A React form's `onSubmit` handler fires twice on every click. "
            "The form is inside a modal dialog. Only the first submission should go through. "
            "Find the root cause and fix it."
        ),
        "tool_outputs": [
            (
                "// components/ModalForm.tsx\n"
                "export function ModalForm({ onClose }) {\n"
                "  const handleSubmit = async (e) => {\n"
                "    await submitData(formData);\n"
                "    onClose();\n"
                "  };\n\n"
                "  return (\n"
                "    <dialog open>\n"
                "      <form onSubmit={handleSubmit}>\n"
                "        <input name='email' />\n"
                "        <button type='submit'>Submit</button>\n"
                "      </form>\n"
                "      <form method='dialog'>\n"
                "        <button>Cancel</button>\n"
                "      </form>\n"
                "    </dialog>\n"
                "  );\n"
                "}\n\n"
                "// Parent component wraps ModalForm inside another <form>\n"
                "// <form onSubmit={parentSubmit}><ModalForm /></form>\n"
            ),
            (
                "# Network tab (Chrome DevTools):\n"
                "POST /api/submit   200 OK   (timestamp: 14:32:01.023)\n"
                "POST /api/submit   200 OK   (timestamp: 14:32:01.025)\n\n"
                "# Both requests have identical payloads.\n"
                "# Console log added to handleSubmit shows it fires once,\n"
                "# but the parent's onSubmit also fires.\n"
                "# The inner form submit event bubbles up to the parent form.\n"
            ),
        ],
    },
    # ── Webpack ──────────────────────────────────────────────────────────
    {
        "task_id": "agentic_017",
        "description": (
            "A Webpack 5 build regressed bundle size from 420 KB to 1.8 MB "
            "after upgrading `lodash` from 4.17.19 to 4.17.21. "
            "The entire lodash library is being bundled instead of individual methods. "
            "Identify the import pattern causing this."
        ),
        "tool_outputs": [
            (
                "$ npx webpack-bundle-analyzer dist/stats.json 2>&1 | head -20\n"
                "# Parsed sizes:\n"
                "  main.js: 1.82 MB\n"
                "    lodash: 1.41 MB  ← whole library\n"
                "    src/: 410 KB\n\n"
                "$ grep -r 'lodash' src/ | grep 'import'\n"
                "src/utils/format.ts:import _ from 'lodash';\n"
                "src/utils/date.ts:import { format, parseISO } from 'date-fns';\n"
                "src/services/array_utils.ts:const chunk = require('lodash/chunk');\n"
                "src/components/Table.tsx:import { sortBy, groupBy } from 'lodash';\n"
            ),
            (
                "$ npx webpack --profile --json > dist/stats.json 2>&1 | tail -5\n"
                "# Module reasons for lodash:\n"
                "  cjs require lodash [./src/utils/format.ts]\n"
                "  harmony import lodash [./src/components/Table.tsx]\n\n"
                "# Both 'import _ from lodash' and 'import { sortBy } from lodash'\n"
                "# pull the full CommonJS build because lodash 4.x ships a UMD\n"
                "# bundle as its main entry, not an ESM build with named exports.\n"
                "# Tree shaking does not work on the CJS build.\n"
            ),
        ],
    },
    # ── FastAPI / Pydantic ───────────────────────────────────────────────
    {
        "task_id": "agentic_018",
        "description": (
            "A FastAPI endpoint raises `ValidationError: value is not a valid dict` "
            "when receiving a JSON body where a nested model field is `null`. "
            "The field should be optional. Fix the Pydantic v2 model definition."
        ),
        "tool_outputs": [
            (
                "# models/order.py\n"
                "from pydantic import BaseModel\n"
                "from typing import Optional\n\n"
                "class Address(BaseModel):\n"
                "    street: str\n"
                "    city: str\n"
                "    country: str = 'US'\n\n"
                "class Order(BaseModel):\n"
                "    id: int\n"
                "    customer_email: str\n"
                "    shipping_address: Optional[Address]  # should allow null\n"
                "    billing_address: Address\n\n"
                "# Router\n"
                "@router.post('/orders')\n"
                "async def create_order(order: Order):\n"
                "    return {'id': order.id}\n"
            ),
            (
                "$ curl -X POST http://localhost:8000/orders \\\n"
                "  -H 'Content-Type: application/json' \\\n"
                "  -d '{\"id\": 1, \"customer_email\": \"a@b.com\", "
                "\"shipping_address\": null, \"billing_address\": {\"street\": "
                "\"123 Main\", \"city\": \"NYC\"}}'\n\n"
                "HTTP/1.1 422 Unprocessable Entity\n"
                "{\"detail\": [{\"type\": \"model_type\", \"loc\": [\"body\", "
                "\"shipping_address\"], \"msg\": \"Input should be a valid dictionary "
                "or instance of Address\", \"input\": null}]}\n\n"
                "# Pydantic v2 changed behavior: Optional[Address] without '= None'\n"
                "# still requires a value. 'null' is not accepted without the default.\n"
            ),
        ],
    },
    # ── pytest ────────────────────────────────────────────────────────────
    {
        "task_id": "agentic_019",
        "description": (
            "Tests pass individually but fail when run together in the same session. "
            "Specifically, `test_create_user` and `test_send_email` conflict. "
            "The issue is a pytest fixture scope problem. Diagnose and fix it."
        ),
        "tool_outputs": [
            (
                "# conftest.py\n"
                "import pytest\n"
                "from myapp import create_app, db\n\n"
                "@pytest.fixture(scope='session')\n"
                "def app():\n"
                "    app = create_app({'TESTING': True, 'SQLALCHEMY_DATABASE_URI': "
                "'sqlite:///:memory:'})\n"
                "    with app.app_context():\n"
                "        db.create_all()\n"
                "        yield app\n\n"
                "@pytest.fixture\n"
                "def client(app):\n"
                "    return app.test_client()\n\n"
                "# test_users.py\n"
                "@pytest.fixture\n"
                "def new_user(client):\n"
                "    client.post('/users', json={'email': 'test@example.com'})\n"
                "    yield\n"
                "    db.session.execute('DELETE FROM user')  # teardown\n"
                "    db.session.commit()\n"
            ),
            (
                "$ pytest tests/ -v 2>&1 | tail -15\n"
                "tests/test_users.py::test_create_user PASSED\n"
                "tests/test_emails.py::test_send_email FAILED\n\n"
                "FAILED tests/test_emails.py::test_send_email - sqlalchemy.exc."
                "InvalidRequestError: This Session's transaction has been rolled back "
                "due to a previous exception during flush.\n\n"
                "$ pytest tests/test_emails.py::test_send_email -v\n"
                "PASSED  ← passes in isolation\n\n"
                "# The session-scoped 'app' fixture shares the DB across tests.\n"
                "# test_create_user's teardown does a raw DELETE that leaves the\n"
                "# SQLAlchemy session in a dirty state for subsequent tests.\n"
            ),
        ],
    },
    # ── Nginx ─────────────────────────────────────────────────────────────
    {
        "task_id": "agentic_020",
        "description": (
            "Nginx returns `502 Bad Gateway` for all requests after deploying "
            "a second upstream server. The single-server configuration worked fine. "
            "The upstream block now has two servers. Diagnose the load balancer config."
        ),
        "tool_outputs": [
            (
                "# /etc/nginx/sites-enabled/app.conf\n"
                "upstream backend {\n"
                "    server 10.0.0.1:8000;\n"
                "    server 10.0.0.2:8000;\n"
                "}\n\n"
                "server {\n"
                "    listen 80;\n"
                "    location / {\n"
                "        proxy_pass http://backend;\n"
                "        proxy_set_header Host $host;\n"
                "        proxy_set_header X-Real-IP $remote_addr;\n"
                "    }\n"
                "}\n\n"
                "# Health check:\n"
                "$ curl http://10.0.0.1:8000/health  → 200 OK\n"
                "$ curl http://10.0.0.2:8000/health  → Connection refused\n"
            ),
            (
                "$ sudo nginx -t\n"
                "nginx: configuration file /etc/nginx/nginx.conf test is successful\n\n"
                "$ sudo tail -20 /var/log/nginx/error.log\n"
                "connect() failed (111: Connection refused) while connecting to upstream, "
                "client: 1.2.3.4, server: _, request: \"GET / HTTP/1.1\", "
                "upstream: \"http://10.0.0.2:8000/\", host: \"example.com\"\n\n"
                "# Default nginx round-robin sends ~50% of requests to 10.0.0.2\n"
                "# which is not yet running. After 2 failed attempts nginx marks\n"
                "# it 'down' but returns 502 for those failed requests.\n"
                "# max_fails and fail_timeout defaults: max_fails=1, fail_timeout=10s\n"
            ),
        ],
    },
]

# Mapping from task_id to list index for fast lookup
TASK_BY_ID = {t["task_id"]: t for t in AGENTIC_TASKS}
