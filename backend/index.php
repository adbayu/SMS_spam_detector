<?php
require __DIR__ . '/bootstrap.php';
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') json_response(['ok' => true]);
$path = trim(parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH), '/');
$body = input_json();
try {
  if ($path === 'api/health') json_response(['ok' => true, 'app' => 'SMS Shield']);


  if ($path === 'api/auth/check' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    $phone = trim($body['phone'] ?? '');
    if ($phone === '') json_response(['error' => 'phone is required'], 422);
    $user = user_by_phone($phone);
    json_response(['exists' => (bool)$user, 'user' => $user]);
  }
  if ($path === 'api/auth/phone' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    $user = upsert_user($body['phone'] ?? '', $body['full_name'] ?? null);
    json_response(['user' => $user]);
  }

  if ($path === 'api/conversations' && $_SERVER['REQUEST_METHOD'] === 'GET') {
    $user = current_user();
    $stmt = db()->prepare('SELECT c.*, m.body last_message, m.prediction last_prediction, m.confidence last_confidence, m.created_at last_message_at FROM conversations c LEFT JOIN messages m ON m.id = (SELECT id FROM messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) WHERE c.user_id = ? ORDER BY c.updated_at DESC');
    $stmt->execute([$user['id']]);
    json_response(['items' => $stmt->fetchAll()]);
  }

  if ($path === 'api/conversations' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    $user = current_user();
    $contactPhone = trim($body['contact_phone'] ?? '');
    if ($contactPhone === '') json_response(['error' => 'contact_phone is required'], 422);
    $stmt = db()->prepare('INSERT INTO conversations (user_id, contact_name, contact_phone, updated_at) VALUES (?, ?, ?, NOW())');
    $stmt->execute([$user['id'], trim($body['contact_name'] ?? 'Unknown'), $contactPhone]);
    json_response(['id' => db()->lastInsertId()], 201);
  }

  if (preg_match('#^api/conversations/(\d+)/messages$#', $path, $m) && $_SERVER['REQUEST_METHOD'] === 'GET') {
    $stmt = db()->prepare('SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC');
    $stmt->execute([(int)$m[1]]);
    json_response(['items' => $stmt->fetchAll()]);
  }

  if (preg_match('#^api/conversations/(\d+)/read$#', $path, $m) && $_SERVER['REQUEST_METHOD'] === 'POST') {
    $user = current_user();
    $stmt = db()->prepare('UPDATE conversations SET unread_count = 0 WHERE id = ? AND user_id = ?');
    $stmt->execute([(int)$m[1], $user['id']]);
    json_response(['ok' => true]);
  }

  if (preg_match('#^api/conversations/(\d+)$#', $path, $m) && $_SERVER['REQUEST_METHOD'] === 'DELETE') {
    $user = current_user();
    $stmt = db()->prepare('DELETE FROM conversations WHERE id = ? AND user_id = ?');
    $stmt->execute([(int)$m[1], $user['id']]);
    json_response(['ok' => true]);
  }

  if ($path === 'api/messages' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    $sender = current_user();
    $message = trim($body['message'] ?? '');
    $recipientPhone = trim($body['contact_phone'] ?? $body['sender_phone'] ?? '');
    $recipientName = trim($body['contact_name'] ?? $recipientPhone);
    if ($message === '' || $recipientPhone === '') json_response(['error' => 'contact_phone and message are required'], 422);

    $recipient = upsert_user($recipientPhone, $recipientName);
    $result = classify_message($message);
    $isSpam = $result['prediction'] === 'spam' ? 1 : 0;

    $getConversation = function (int $userId, string $contactPhone, string $contactName) {
      $find = db()->prepare('SELECT * FROM conversations WHERE user_id = ? AND contact_phone = ? LIMIT 1');
      $find->execute([$userId, $contactPhone]);
      $conversation = $find->fetch();
      if ($conversation) return (int)$conversation['id'];
      $create = db()->prepare('INSERT INTO conversations (user_id, contact_name, contact_phone, updated_at) VALUES (?, ?, ?, NOW())');
      $create->execute([$userId, $contactName, $contactPhone]);
      return (int)db()->lastInsertId();
    };

    $recipientConversationId = $getConversation((int)$recipient['id'], $sender['phone'], $sender['full_name']);
    $insert = db()->prepare('INSERT INTO messages (conversation_id, sender_phone, body, direction, prediction, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, NOW())');
    $insert->execute([$recipientConversationId, $sender['phone'], $message, 'incoming', $result['prediction'], $result['confidence']]);
    $recipientMessageId = db()->lastInsertId();
    db()->prepare('UPDATE conversations SET contact_name = ?, is_spam = GREATEST(is_spam, ?), unread_count = unread_count + 1, updated_at = NOW() WHERE id = ?')->execute([$sender['full_name'], $isSpam, $recipientConversationId]);

    if ($isSpam) {
      $explain = json_encode(['keywords' => [], 'indicators' => ['Model classified this SMS as spam'], 'confidence' => $result['confidence']]);
      db()->prepare('INSERT INTO spam_analysis (message_id, severity, explanation, created_at) VALUES (?, ?, ?, NOW())')->execute([$recipientMessageId, severity_from_confidence((float)$result['confidence']), $explain]);

      $senderConversationId = $getConversation((int)$sender['id'], $recipientPhone, $recipientName);
      $hiddenBody = 'Pesan disembunyikan karena terdeteksi spam.';
      $insert->execute([$senderConversationId, $sender['phone'], $hiddenBody, 'outgoing', 'spam', $result['confidence']]);
      $senderMessageId = db()->lastInsertId();
      db()->prepare('UPDATE conversations SET contact_name = ?, is_spam = 1, updated_at = NOW() WHERE id = ?')->execute([$recipientName, $senderConversationId]);
      db()->prepare('INSERT INTO spam_analysis (message_id, severity, explanation, created_at) VALUES (?, ?, ?, NOW())')->execute([$senderMessageId, severity_from_confidence((float)$result['confidence']), $explain]);
      json_response(['id' => $senderMessageId, 'conversation_id' => $senderConversationId, 'recipient_conversation_id' => $recipientConversationId, 'hidden_from_sender' => true, 'classification' => $result], 201);
    }

    $senderConversationId = $getConversation((int)$sender['id'], $recipientPhone, $recipientName);
    $insert->execute([$senderConversationId, $sender['phone'], $message, 'outgoing', 'ham', $result['confidence']]);
    $senderMessageId = db()->lastInsertId();
    db()->prepare('UPDATE conversations SET contact_name = ?, is_spam = 0, updated_at = NOW() WHERE id = ?')->execute([$recipientName, $senderConversationId]);

    json_response(['id' => $senderMessageId, 'conversation_id' => $senderConversationId, 'recipient_conversation_id' => $recipientConversationId, 'classification' => $result], 201);
  }

  if ($path === 'api/spam' && $_SERVER['REQUEST_METHOD'] === 'GET') {
    $user = current_user();
    $stmt = db()->prepare("SELECT m.id message_id, m.conversation_id, m.sender_phone, m.body, m.type, m.direction, m.status, m.prediction, m.confidence, m.created_at, c.contact_name, c.contact_phone, s.country, s.carrier, s.severity, s.explanation FROM messages m JOIN conversations c ON c.id = m.conversation_id LEFT JOIN spam_analysis s ON s.message_id = m.id WHERE c.user_id = ? AND m.prediction = 'spam' ORDER BY m.created_at DESC");
    $stmt->execute([$user['id']]);
    json_response(['items' => $stmt->fetchAll()]);
  }
  json_response(['error' => 'Not found'], 404);
} catch (Throwable $e) {
  json_response(['error' => $e->getMessage()], 500);
}






