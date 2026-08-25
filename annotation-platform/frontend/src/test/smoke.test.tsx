/// <reference types="vitest/globals" />

import { render, screen } from '@testing-library/react';

describe('smoke', () => {
  it('vitest runs', () => {
    expect(1 + 1).toBe(2);
  });

  it('renders hello text', () => {
    render(<div>Hello Labeling Platform</div>);
    expect(screen.getByText('Hello Labeling Platform')).toBeInTheDocument();
  });
});
