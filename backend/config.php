<?php
return [
  'db' => [
    'dsn' => getenv('DB_DSN') ?: 'mysql:host=127.0.0.1;dbname=sms_shield;charset=utf8mb4',
    'user' => getenv('DB_USER') ?: 'root',
    'pass' => getenv('DB_PASS') ?: '',
  ],
  'python' => getenv('PYTHON_BIN') ?: 'python',
  'model_script' => getenv('SPAM_MODEL_SCRIPT') ?: dirname(__DIR__) . DIRECTORY_SEPARATOR . 'models' . DIRECTORY_SEPARATOR . 'src' . DIRECTORY_SEPARATOR . 'predict.py',
];
