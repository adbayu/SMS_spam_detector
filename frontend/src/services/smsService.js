const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.phone ? { 'X-User-Phone': options.phone } : {}),
      ...(options.headers || {}),
    },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || 'API request failed');
  return data;
}

export function checkPhone(phone) { return request('/api/auth/check', { method: 'POST', body: JSON.stringify({ phone }) }); }
export function loginWithPhone(phone, fullName) {
  return request('/api/auth/phone', { method: 'POST', body: JSON.stringify({ phone, full_name: fullName }) });
}
export function getConversations(phone) { return request('/api/conversations', { phone }); }
export function getMessages(phone, conversationId) { return request(`/api/conversations/${conversationId}/messages`, { phone }); }
export function getSpamMessages(phone) { return request('/api/spam', { phone }); }
export function markConversationRead(phone, conversationId) { return request(`/api/conversations/${conversationId}/read`, { method: 'POST', phone }); }
export function unblockConversation(phone, conversationId) { return request(`/api/conversations/${conversationId}/unblock`, { method: 'POST', phone }); }
export function deleteConversation(phone, conversationId) { return request(`/api/conversations/${conversationId}`, { method: 'DELETE', phone }); }
export function deleteMessage(phone, messageId) { return request(`/api/messages/${messageId}`, { method: 'DELETE', phone }); }
export function sendMessage(phone, payload) {
  return request('/api/messages', { method: 'POST', phone, body: JSON.stringify(payload) });
}

