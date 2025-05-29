// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app.py', // Asegúrate de que apunte a tu archivo principal de Streamlit
    // Si tienes más archivos .py en subcarpetas (como 'pages') que usan clases, añádelos:
    // './pages/**/*.py',
  ],
  theme: {
    extend: { // Importante usar 'extend' para no sobrescribir los colores base de Tailwind
      colors: {
        'fondo-principal': '#F0F2F6',
        'primario-app': '#0A3D62',
        'acento-app': '#1E88E5',
        'texto-principal': '#333333',
        'texto-secundario': '#555555',
        'feedback-info-bg': '#E0F2FE', // Para el fondo del mensaje de IA
        'feedback-info-text': '#0C4A6E', // Para el texto del mensaje de IA
        // Puedes añadir más colores si los necesitas
        'peligro-app': '#D32F2F', // Ejemplo para errores o alertas
        'exito-app': '#388E3C',   // Ejemplo para mensajes de éxito
        'test-debug-color': '#FF00FF',
      },
      fontFamily: {
        // Si quieres usar una fuente específica y la tienes configurada
        // sans: ['Inter', 'ui-sans-serif', 'system-ui'], // Ejemplo con Inter como principal
      },
    },
  },
  plugins: [
    // Aquí podrías añadir plugins de Tailwind si los necesitaras, por ejemplo, @tailwindcss/forms
    // require('@tailwindcss/forms'), // Esto requeriría `npm install @tailwindcss/forms`
    // Por ahora, lo dejamos vacío.
  ],
}