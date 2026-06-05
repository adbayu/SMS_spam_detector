<?php
function json_response($data, int $status = 200): void {
  http_response_code($status);
  header('Content-Type: application/json');
  header('Access-Control-Allow-Origin: *');
  header('Access-Control-Allow-Headers: Content-Type, X-User-Phone');
  header('Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS');
  echo json_encode($data, JSON_UNESCAPED_SLASHES);
  exit;
}
function db(): PDO {
  static $pdo;
  if ($pdo) return $pdo;
  $config = require __DIR__ . '/config.php';
  $pdo = new PDO($config['db']['dsn'], $config['db']['user'], $config['db']['pass'], [
    PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
  ]);
  return $pdo;
}
function input_json(): array {
  return json_decode(file_get_contents('php://input') ?: '{}', true) ?: [];
}
function user_by_phone(string $phone): ?array {
  $stmt = db()->prepare('SELECT * FROM users WHERE phone = ? LIMIT 1');
  $stmt->execute([$phone]);
  $user = $stmt->fetch();
  return $user ?: null;
}
function upsert_user(string $phone, ?string $name = null): array {
  $phone = trim($phone);
  if ($phone === '') json_response(['error' => 'phone is required'], 422);
  $user = user_by_phone($phone);
  if ($user) return $user;
  $fullName = $name ?: 'User ' . substr(preg_replace('/\D+/', '', $phone), -4);
  $stmt = db()->prepare('INSERT INTO users (full_name, phone, created_at) VALUES (?, ?, NOW())');
  $stmt->execute([$fullName, $phone]);
  return user_by_phone($phone);
}
function current_user(): array {
  $phone = $_SERVER['HTTP_X_USER_PHONE'] ?? $_GET['phone'] ?? '';
  return upsert_user($phone);
}
function normalize_prediction(string $label): string {
  return strtolower(trim($label)) === 'spam' ? 'spam' : 'ham';
}
function classify_message(string $message): array {
  $config = require __DIR__ . '/config.php';
  $script = $config['model_script'];
  if (!is_file($script)) json_response(['error' => 'model script not found', 'path' => $script], 500);
  $cmd = escapeshellarg($config['python']) . ' ' . escapeshellarg($script) . ' ' . escapeshellarg($message);
  $lines = [];
  $code = 0;
  exec($cmd . ' 2>&1', $lines, $code);
  $raw = trim(implode("\n", $lines));
  if ($code !== 0) json_response(['error' => 'model prediction failed', 'detail' => $raw], 500);
  $decoded = json_decode($raw, true);
  if (is_array($decoded) && isset($decoded['prediction'])) {
    return [
      'message' => $message,
      'prediction' => normalize_prediction($decoded['prediction']),
      'confidence' => isset($decoded['confidence']) ? (float)$decoded['confidence'] : 0.9,
    ];
  }
  return ['message' => $message, 'prediction' => normalize_prediction($raw), 'confidence' => 0.9];
}
function severity_from_confidence(float $confidence): string {
  if ($confidence >= 0.9) return 'high';
  if ($confidence >= 0.75) return 'medium';
  return 'low';
}
