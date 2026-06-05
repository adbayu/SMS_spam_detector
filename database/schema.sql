CREATE DATABASE IF NOT EXISTS sms_shield CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE sms_shield;

CREATE TABLE users (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  full_name VARCHAR(120) NOT NULL,
  phone VARCHAR(32) NOT NULL UNIQUE,
  email VARCHAR(160) NULL,
  avatar_url VARCHAR(255) NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversations (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  user_id BIGINT UNSIGNED NOT NULL,
  contact_name VARCHAR(120) NOT NULL,
  contact_phone VARCHAR(32) NOT NULL,
  is_spam BOOLEAN DEFAULT FALSE,
  unread_count INT DEFAULT 0,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_user_updated (user_id, updated_at),
  UNIQUE KEY uniq_user_contact (user_id, contact_phone)
);

CREATE TABLE messages (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  conversation_id BIGINT UNSIGNED NOT NULL,
  sender_phone VARCHAR(32) NOT NULL,
  body TEXT NOT NULL,
  type ENUM('text','image','attachment','link') DEFAULT 'text',
  direction ENUM('incoming','outgoing') DEFAULT 'incoming',
  status ENUM('sent','delivered','read') NULL,
  prediction ENUM('ham','spam') DEFAULT 'ham',
  confidence DECIMAL(5,4) DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
  INDEX idx_conversation_created (conversation_id, created_at),
  INDEX idx_prediction_created (prediction, created_at)
);

CREATE TABLE spam_analysis (
  id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  message_id BIGINT UNSIGNED NOT NULL,
  country VARCHAR(80) NULL,
  carrier VARCHAR(120) NULL,
  severity ENUM('low','medium','high') DEFAULT 'low',
  explanation JSON NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);
