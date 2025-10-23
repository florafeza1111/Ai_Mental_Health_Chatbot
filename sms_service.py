"""
SMS Service for AIMHSA
Integrates with HDEV SMS Gateway for sending notifications
"""

import requests
import json
import time
from typing import Dict, Optional, Tuple
import logging

class HDevSMSService:
    def __init__(self, api_id: str, api_key: str):
        """
        Initialize HDEV SMS Service
        
        Args:
            api_id: HDEV API ID
            api_key: HDEV API Key
        """
        self.api_id = api_id
        self.api_key = api_key
        self.base_url = "https://sms-api.hdev.rw/v1/api"
        self.logger = logging.getLogger(__name__)
    
    def send_sms(self, sender_id: str, phone_number: str, message: str, link: str = '') -> Dict:
        """
        Send SMS message
        
        Args:
            sender_id: Sender identifier (max 11 characters)
            phone_number: Recipient phone number (Rwanda format: +250XXXXXXXXX)
            message: SMS message content
            link: Optional link to include
            
        Returns:
            Dict with response status and details
        """
        try:
            # Format phone number for Rwanda
            formatted_phone = self._format_phone_number(phone_number)
            
            # Prepare request data
            data = {
                'ref': 'sms',
                'sender_id': 'N-SMS',  # Use N-SMS as sender ID
                'tel': formatted_phone,
                'message': message,
                'link': link
            }
            
            # Make API request
            url = f"{self.base_url}/{self.api_id}/{self.api_key}"
            response = requests.post(url, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"SMS sent successfully to {formatted_phone}: {result}")
                return {
                    'success': True,
                    'response': result,
                    'phone': formatted_phone
                }
            else:
                self.logger.error(f"SMS API error: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"API Error: {response.status_code}",
                    'phone': formatted_phone
                }
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"SMS request failed: {str(e)}")
            return {
                'success': False,
                'error': f"Request failed: {str(e)}",
                'phone': phone_number
            }
        except Exception as e:
            self.logger.error(f"SMS service error: {str(e)}")
            return {
                'success': False,
                'error': f"Service error: {str(e)}",
                'phone': phone_number
            }
    
    def send_booking_notification(self, user_data: Dict, professional_data: Dict, booking_data: Dict) -> Dict:
        """
        Send booking notification SMS to user
        
        Args:
            user_data: User information (name, phone, etc.)
            professional_data: Professional information
            booking_data: Booking details
            
        Returns:
            Dict with SMS sending result
        """
        try:
            # Format user name
            user_name = user_data.get('fullname', user_data.get('username', 'User'))
            
            # Format professional name
            prof_name = f"{professional_data.get('first_name', '')} {professional_data.get('last_name', '')}".strip()
            prof_specialization = professional_data.get('specialization', 'Mental Health Professional')
            
            # Format scheduled time
            scheduled_time = self._format_datetime(booking_data.get('scheduled_time', 0))
            
            # Create message based on risk level
            risk_level = booking_data.get('risk_level', 'medium')
            session_type = booking_data.get('session_type', 'consultation')
            
            if risk_level == 'critical':
                urgency_text = "URGENT: Emergency mental health support has been arranged"
            elif risk_level == 'high':
                urgency_text = "URGENT: Professional mental health support has been scheduled"
            else:
                urgency_text = "Professional mental health support has been scheduled"
            
            # Build SMS message
            message = f"""AIMHSA Mental Health Support

{urgency_text}

Professional: {prof_name}
Specialization: {prof_specialization}
Scheduled: {scheduled_time}
Session Type: {session_type.title()}

You will be contacted shortly. If this is an emergency, call 112 or the Mental Health Hotline at 105.

Stay safe and take care.
AIMHSA Team"""
            
            # Send SMS
            return self.send_sms(
                sender_id="N-SMS",
                phone_number=user_data.get('telephone', ''),
                message=message
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send booking notification: {str(e)}")
            return {
                'success': False,
                'error': f"Notification failed: {str(e)}"
            }
    
    def send_professional_notification(self, professional_data: Dict, user_data: Dict, booking_data: Dict) -> Dict:
        """
        Send notification SMS to professional about new booking
        
        Args:
            professional_data: Professional information
            user_data: User information
            booking_data: Booking details
            
        Returns:
            Dict with SMS sending result
        """
        try:
            # Format names
            prof_name = f"{professional_data.get('first_name', '')} {professional_data.get('last_name', '')}".strip()
            user_name = user_data.get('fullname', user_data.get('username', 'User'))
            
            # Format scheduled time
            scheduled_time = self._format_datetime(booking_data.get('scheduled_time', 0))
            
            # Build message with comprehensive user information
            risk_level = booking_data.get('risk_level', 'medium')
            booking_id = booking_data.get('booking_id', 'N/A')
            
            # Get user contact information
            user_phone = user_data.get('telephone', 'Not provided')
            user_email = user_data.get('email', 'Not provided')
            user_location = f"{user_data.get('district', 'Unknown')}, {user_data.get('province', 'Unknown')}"
            
            message = f"""AIMHSA Professional Alert

New {risk_level.upper()} risk booking assigned to you.

Booking ID: {booking_id}
User: {user_name}
Risk Level: {risk_level.upper()}
Scheduled: {scheduled_time}

USER CONTACT INFORMATION:
Phone: {user_phone}
Email: {user_email}
Location: {user_location}

Please check your professional dashboard for details and accept/decline the booking.

AIMHSA System"""
            
            # Send SMS to professional
            return self.send_sms(
                sender_id="N-SMS",
                phone_number=professional_data.get('phone', ''),
                message=message
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send professional notification: {str(e)}")
            return {
                'success': False,
                'error': f"Professional notification failed: {str(e)}"
            }
    
    def _format_phone_number(self, phone: str) -> str:
        """
        Format phone number for Rwanda SMS
        
        Args:
            phone: Phone number in various formats
            
        Returns:
            Formatted phone number (+250XXXXXXXXX)
        """
        if not phone:
            return ""
        
        # Remove all non-digit characters
        digits = ''.join(filter(str.isdigit, phone))
        
        # Handle different formats
        if digits.startswith('250'):
            return f"+{digits}"
        elif digits.startswith('0'):
            return f"+250{digits[1:]}"
        elif len(digits) == 9:
            return f"+250{digits}"
        else:
            return f"+{digits}"
    
    def _format_datetime(self, timestamp: float) -> str:
        """
        Format timestamp to readable datetime
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Formatted datetime string
        """
        try:
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return "TBD"
    
    def test_connection(self) -> bool:
        """
        Test SMS service connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Send a test SMS to a dummy number
            result = self.send_sms(
                sender_id="N-SMS",
                phone_number="+250000000000",  # Dummy number
                message="Test message"
            )
            return result.get('success', False)
        except:
            return False

# Global SMS service instance
sms_service = None

def initialize_sms_service(api_id: str, api_key: str):
    """Initialize the global SMS service"""
    global sms_service
    sms_service = HDevSMSService(api_id, api_key)
    return sms_service

def get_sms_service() -> Optional[HDevSMSService]:
    """Get the global SMS service instance"""
    return sms_service

