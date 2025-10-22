// Main JavaScript for ML Flask Portfolio

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Enable Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert.classList.contains('show')) {
                bootstrap.Alert.getInstance(alert).close();
            }
        }, 5000);
    });
});

// Utility functions
const MLApp = {
    // Format percentage
    formatPercent: (value) => {
        return (value * 100).toFixed(1) + '%';
    },
    
    // Format number with precision
    formatNumber: (value, precision = 4) => {
        return parseFloat(value).toFixed(precision);
    },
    
    // Show loading state
    showLoading: (element) => {
        element.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>';
    },
    
    // Handle API errors
    handleApiError: (error) => {
        console.error('API Error:', error);
        return 'An error occurred while processing your request.';
    },
    
    // Validate numeric input
    validateNumeric: (value, min = null, max = null) => {
        const num = parseFloat(value);
        if (isNaN(num)) return false;
        if (min !== null && num < min) return false;
        if (max !== null && num > max) return false;
        return true;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLApp;
}