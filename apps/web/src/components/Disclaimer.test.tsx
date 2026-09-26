import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { Disclaimer } from './Disclaimer.js';

describe('Disclaimer', () => {
  it('states the prototype is not for clinical use', () => {
    render(<Disclaimer />);
    expect(screen.getByRole('note')).toHaveTextContent(/not a medical device/i);
    expect(screen.getByRole('note')).toHaveTextContent(/must not be used for clinical/i);
  });
});
