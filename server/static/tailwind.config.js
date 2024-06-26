/** @type {import('tailwindcss').Config} */
module.exports = {

  content: ['../templates/*.{html,js}',
            'css/*.css',
            'script/*.js',
            'node_modules/flowbite/**/*.js'],
  theme: {
    extend: {
      colors: {
        primary: '#202225',
        secondary: '#5865f2'
      }
    },
  },
  plugins: [
    require('tailwindcss'),
    require('autoprefixer'),
    require('flowbite/plugin')
  ],
}

