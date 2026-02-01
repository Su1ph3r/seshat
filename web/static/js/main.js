/**
 * Seshat Web Interface - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    // Alert close buttons
    document.querySelectorAll('.alert-close').forEach(function(btn) {
        btn.addEventListener('click', function() {
            this.parentElement.remove();
        });
    });

    // Auto-hide alerts after 5 seconds
    document.querySelectorAll('.alert').forEach(function(alert) {
        setTimeout(function() {
            alert.style.opacity = '0';
            alert.style.transition = 'opacity 0.3s';
            setTimeout(function() {
                alert.remove();
            }, 300);
        }, 5000);
    });

    // Form validation
    document.querySelectorAll('form').forEach(function(form) {
        form.addEventListener('submit', function(e) {
            const textareas = form.querySelectorAll('textarea[required]');
            let valid = true;

            textareas.forEach(function(textarea) {
                if (textarea.value.trim().length < 10) {
                    valid = false;
                    textarea.classList.add('error');
                    showError(textarea, 'Please enter at least 10 characters');
                } else {
                    textarea.classList.remove('error');
                    hideError(textarea);
                }
            });

            if (!valid) {
                e.preventDefault();
            }
        });
    });

    // Character count for textareas
    document.querySelectorAll('textarea').forEach(function(textarea) {
        const wrapper = document.createElement('div');
        wrapper.className = 'textarea-wrapper';
        textarea.parentNode.insertBefore(wrapper, textarea);
        wrapper.appendChild(textarea);

        const counter = document.createElement('div');
        counter.className = 'char-counter';
        counter.style.cssText = 'text-align: right; font-size: 0.8rem; color: #6c757d; margin-top: 0.25rem;';
        wrapper.appendChild(counter);

        function updateCounter() {
            const text = textarea.value;
            const words = text.trim() ? text.trim().split(/\s+/).length : 0;
            const chars = text.length;
            counter.textContent = `${words} words, ${chars} characters`;
        }

        textarea.addEventListener('input', updateCounter);
        updateCounter();
    });

    // File upload preview
    document.querySelectorAll('input[type="file"]').forEach(function(input) {
        input.addEventListener('change', function() {
            const fileList = this.files;
            if (fileList.length > 0) {
                const names = Array.from(fileList).map(f => f.name).join(', ');
                let preview = this.parentNode.querySelector('.file-preview');
                if (!preview) {
                    preview = document.createElement('div');
                    preview.className = 'file-preview';
                    preview.style.cssText = 'margin-top: 0.5rem; font-size: 0.9rem; color: #4a90d9;';
                    this.parentNode.appendChild(preview);
                }
                preview.textContent = 'Selected: ' + names;
            }
        });
    });

    // Confirmation dialogs
    document.querySelectorAll('[data-confirm]').forEach(function(el) {
        el.addEventListener('click', function(e) {
            if (!confirm(this.dataset.confirm)) {
                e.preventDefault();
            }
        });
    });

    // Copy to clipboard functionality
    document.querySelectorAll('.copy-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            const target = document.querySelector(this.dataset.target);
            if (target) {
                navigator.clipboard.writeText(target.textContent).then(function() {
                    btn.textContent = 'Copied!';
                    setTimeout(function() {
                        btn.textContent = 'Copy';
                    }, 2000);
                });
            }
        });
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Auto-expand textareas
    document.querySelectorAll('textarea').forEach(function(textarea) {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 500) + 'px';
        });
    });
});

// Error handling helpers
function showError(element, message) {
    let error = element.parentNode.querySelector('.error-message');
    if (!error) {
        error = document.createElement('div');
        error.className = 'error-message';
        error.style.cssText = 'color: #dc3545; font-size: 0.85rem; margin-top: 0.25rem;';
        element.parentNode.appendChild(error);
    }
    error.textContent = message;
}

function hideError(element) {
    const error = element.parentNode.querySelector('.error-message');
    if (error) {
        error.remove();
    }
}

// Loading state helper
function setLoading(form, loading) {
    const btn = form.querySelector('button[type="submit"]');
    if (loading) {
        btn.disabled = true;
        btn.dataset.originalText = btn.textContent;
        btn.textContent = 'Processing...';
    } else {
        btn.disabled = false;
        btn.textContent = btn.dataset.originalText || 'Submit';
    }
}

// API helper functions
const API = {
    async analyze(text) {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        return response.json();
    },

    async compare(text1, text2) {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text1, text2 })
        });
        return response.json();
    },

    async getProfiles() {
        const response = await fetch('/api/profiles');
        return response.json();
    }
};

// Chart helpers
function createRadarChart(containerId, labels, data, title) {
    if (typeof Plotly === 'undefined') return;

    const trace = {
        type: 'scatterpolar',
        r: data.concat([data[0]]),
        theta: labels.concat([labels[0]]),
        fill: 'toself',
        fillcolor: 'rgba(74, 144, 217, 0.3)',
        line: { color: 'rgba(74, 144, 217, 1)' }
    };

    const layout = {
        polar: {
            radialaxis: {
                visible: true,
                range: [0, 1]
            }
        },
        title: title,
        showlegend: false
    };

    Plotly.newPlot(containerId, [trace], layout);
}

function createBarChart(containerId, labels, values, title) {
    if (typeof Plotly === 'undefined') return;

    const trace = {
        type: 'bar',
        x: labels,
        y: values,
        marker: { color: '#4a90d9' }
    };

    const layout = {
        title: title,
        xaxis: { tickangle: -45 },
        margin: { b: 120 }
    };

    Plotly.newPlot(containerId, [trace], layout);
}
