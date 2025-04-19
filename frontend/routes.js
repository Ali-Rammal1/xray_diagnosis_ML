// This file is used for reference but actual routing is implemented in App.jsx
const routes = [
    {
        path: '/',
        name: 'Home',
        exact: true,
    },
    {
        path: '/results',
        name: 'Results',
        exact: true,
    },
    {
        path: '*',
        name: '404',
    }
];

export default routes;